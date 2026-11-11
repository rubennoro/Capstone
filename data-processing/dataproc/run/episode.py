"""One recording of ``dataproc run``: stage it, run the stages, then promote it, fail it or keep it waiting.

::

    rp = RecordingProcessor(root, ctx, log)          # ctx: the folder's ProcessContext with its calibration
    outcome = rp.run(recording)                       # a plan.Recording
    outcome.state, outcome.slot, outcome.describe()

* ``take``: adapt the Pi recording into ``incoming/``, intake it into ``tasks/<folder>/.staging/<pi_name>/raw``
  (recorded in ``pi_sources.csv``); then process it as below. A recording intake quarantines (a duplicate) is
  ``failed`` at stage ``intake`` and stays in ``quarantine/``. One that cannot be converted is skipped and not
  recorded (it can be retried).
* ``resume``: process ``.staging/<pi_name>`` as it is (stages already done are not redone).
* ``retry``: move ``failed/<pi_name>`` back to ``.staging/`` first.
* ``reprocess`` (``--force``): re-run the stages of an existing ``episode_N`` in place.

Then the :class:`~dataproc.process.EpisodeProcessor` stages run, and:

* passed (export ``done``, ``processed/episode.zarr`` written) -> ``episode_N`` (the next free number);
* stopped at slam because no SLAM backend is installed -> stays in ``.staging/``: ``waiting``;
* failed or rejected at any stage -> ``failed/<pi_name>/`` with ``reason.json``; ``pi_sources.csv`` status
  ``failed``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dataproc.files import load_status
from dataproc.paths import STAGING_DIR, DataRoot
from dataproc.report import episode_summary
from dataproc.run import staging
from dataproc.run.plan import REPROCESS, RESUME, RETRY, TAKE, Recording

__all__ = ["EPISODE", "FAILED", "WAITING", "SKIPPED", "Outcome", "RecordingProcessor"]

EPISODE, FAILED, WAITING, SKIPPED = "episode", "failed", "waiting", "skipped"


@dataclass
class Outcome:
    """What happened to one recording. ``state``: ``episode`` (``slot`` = ``episode_N``), ``failed`` (``stage``,
    ``reasons``, ``message``; ``slot`` = ``failed/<pi_name>``), ``waiting`` (for SLAM; ``slot`` =
    ``.staging/<pi_name>``) or ``skipped`` (``message`` says why)."""

    name: str
    pi_name: str | None
    state: str
    slot: str | None = None
    stage: str | None = None
    reasons: list[str] = field(default_factory=list)
    message: str = ""
    action: str = ""
    ran: list[str] = field(default_factory=list)
    quiet: bool = False
    seconds: float = 0.0

    def describe(self) -> str:
        if self.state == EPISODE:
            return str(self.slot)
        if self.state == FAILED:
            return f"failed ({self.stage}: {'; '.join(self.reasons) or self.message})"
        if self.state == WAITING:
            return "waiting for SLAM"
        return f"skipped ({self.message})"

    def to_dict(self) -> dict:
        return {"source_name": self.name, "pi_name": self.pi_name, "outcome": self.state, "slot": self.slot,
                "stage": self.stage, "reasons": list(self.reasons), "message": self.message, "action": self.action,
                "ran": list(self.ran), "seconds": round(self.seconds, 3)}


class RecordingProcessor:
    """Processes the recordings of one run, one at a time (see the module docstring). Raises for problems that
    stop the run (intake, filesystem); every problem of the recording itself is its :class:`Outcome`."""

    def __init__(self, root: DataRoot, ctx, log: Callable = print):
        self.root, self.ctx, self.log = root, ctx, log
        self.task_dir: Path = ctx.task_dir

    def run(self, rec: Recording, force: bool = False) -> Outcome:
        t0 = time.perf_counter()
        out = self._run(rec, force)
        out.action, out.seconds = rec.action, time.perf_counter() - t0
        return out

    def _run(self, rec: Recording, force: bool) -> Outcome:
        if rec.action == TAKE:
            staged = self.take(rec)
            if isinstance(staged, Outcome):
                return staged
        elif rec.action == RESUME:
            self.record_source(rec)
        elif rec.action == RETRY:
            staging.unfail(self.task_dir, rec.pi_name)
            staging.sync_sources(self.root, self.task_dir)
        elif rec.action == REPROCESS:
            return self.process(rec, rec.slot, force=True)
        else:
            raise ValueError(f"nothing to do for {rec.name} ({rec.action})")
        return self.process(rec, f"{STAGING_DIR}/{rec.pi_name}", force=force)

    # ----------------------------------------------------------------------------------------- steps

    def take(self, rec: Recording) -> Outcome | None:
        """Adapt + intake into ``.staging/<pi_name>``. Returns an :class:`Outcome` when it did not get there."""
        from dataproc.pi_intake import adapt_sources, intake_adapted

        adapted, problems = adapt_sources(self.root, [rec.source], self.ctx.task.rig)
        if problems:
            return Outcome(rec.name, rec.pi_name, SKIPPED, message=problems[rec.name])
        taken = intake_adapted(self.root, self.ctx.folder, adapted, "demonstration", staging=True,
                               log=lambda *a, **k: None)
        o = taken.outcomes.get(rec.name)
        if o is None:
            return Outcome(rec.name, rec.pi_name, SKIPPED, message="intake did not take it in")
        rec.pi_name = o.pi_name
        if o.status == "quarantined":
            where = o.quarantine_dir.relative_to(self.root.path) if o.quarantine_dir else "quarantine/"
            return Outcome(rec.name, o.pi_name, FAILED, slot=str(where), stage="intake", reasons=list(o.reasons),
                           message=f"quarantined by intake (see {where}/reason.json)")
        if o.status != "ingested":
            return Outcome(rec.name, o.pi_name, SKIPPED, message=f"intake: {o.describe()}")
        return None

    def record_source(self, rec: Recording) -> None:
        """A ``.staging/`` recording whose ``pi_sources.csv`` row was lost (a crash right after intake): add it."""
        from dataproc.pi_adapter import SOURCES_FILE, append_sources

        src = rec.source
        if src is None or rec.taken is not None:
            return
        append_sources(self.root.path / SOURCES_FILE, [{
            "source_name": src.name, "source_manifest_sha256": src.manifest_sha256, "converted_name": rec.pi_name,
            "task": self.ctx.folder, "kind": "demonstration", "status": "ingested",
            "adapted_unix_ns": time.time_ns()}])
        self.log(f"reconciled: {rec.name} was in .staging/{rec.pi_name} but not in {SOURCES_FILE}; recorded it")

    def process(self, rec: Recording, slot: str, force: bool) -> Outcome:
        """Run the stages on ``task_dir/<slot>``, then move it to where it belongs."""
        from dataproc.process import EpisodeProcessor

        ep_dir = self.task_dir / slot
        self.ctx.force = force
        try:
            ran = EpisodeProcessor(self.ctx, ep_dir, pi_name=rec.pi_name).run()
        finally:
            self.ctx.force = False
        status = load_status(ep_dir / "processed")
        out = Outcome(rec.name, rec.pi_name, EPISODE, slot=slot, ran=ran)
        if (episode_summary(status)["status"] == "exported"
                and (ep_dir / "processed" / "episode.zarr").is_dir()):
            if slot.startswith(f"{STAGING_DIR}/"):
                out.slot = staging.promote(self.task_dir, rec.pi_name)
        elif staging.waiting_for_slam(status) and slot.startswith(f"{STAGING_DIR}/"):
            out.state = WAITING
        else:
            reason = staging.fail(self.task_dir, slot, rec.pi_name, rec.source_name)
            out.state, out.slot = FAILED, f"failed/{rec.pi_name}"
            out.stage, out.reasons, out.message = reason["stage"], reason["reasons"], reason["message"]
        staging.sync_sources(self.root, self.task_dir)
        return out
