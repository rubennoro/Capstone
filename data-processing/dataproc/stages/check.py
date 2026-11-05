"""Stage ``check``: the raw-episode contract plus the rig consistency rules (docs/DESIGN.md C1, docs/PROCESS.md).

* every :meth:`RawEpisode.validate` error (``decode_videos=True``, manifest checked) is a reason, in report order;
* ``rig_resolution_mismatch`` / ``rig_fps_mismatch``: a ``meta.json`` camera ``resolution`` / ``fps`` differs from
  ``rigs/<rig>.yaml`` for the camera with that role;
* ``rig_mismatch``: ``meta.json`` ``rig_id`` differs from ``task.yaml`` ``rig``.

Any reason rejects the episode. Warnings (sequence gaps, button outside a stream) are kept as details only: the
validate stage judges gaps with its own thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dataproc.files import write_json
from dataproc.stages.base import Stage, StageFailed

__all__ = ["FPS_TOLERANCE", "CheckResult", "rig_consistency", "check_episode", "CheckStage"]

#: meta.json fps vs rig yaml fps: equal within this (both are nominal rates).
FPS_TOLERANCE = 1e-6


@dataclass
class CheckResult:
    reasons: list[str]
    details: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.reasons


def rig_consistency(meta, rig, task_rig: str) -> tuple[list[str], dict]:
    """C1 reasons (``rig_resolution_mismatch``, ``rig_fps_mismatch``, ``rig_mismatch``) and their details."""
    reasons, details = [], {}
    for cam, cm in sorted(meta.cameras.items()):
        rc = rig.cameras.get(cm.role)
        if rc is None:          # the validator already reports camera_roles
            continue
        if tuple(cm.resolution) != tuple(rc.resolution):
            details.setdefault("rig_resolution_mismatch", []).append(
                {"camera": cam, "role": cm.role, "meta": list(cm.resolution), "rig": list(rc.resolution)})
        if abs(float(cm.fps) - float(rc.fps)) > FPS_TOLERANCE:
            details.setdefault("rig_fps_mismatch", []).append(
                {"camera": cam, "role": cm.role, "meta": float(cm.fps), "rig": float(rc.fps)})
    for code in ("rig_resolution_mismatch", "rig_fps_mismatch"):
        if code in details:
            reasons.append(code)
    if meta.rig_id != task_rig:
        reasons.append("rig_mismatch")
        details["rig_mismatch"] = {"episode_rig": meta.rig_id, "task_rig": task_rig}
    return reasons, details


def check_episode(raw, rig, task_rig: str) -> CheckResult:
    """Run the contract validator and C1 on an opened :class:`~dataproc.raw_episode.RawEpisode`."""
    report = raw.validate(decode_videos=True, check_manifest=True)
    reasons: list[str] = []
    for p in report.errors:
        if p.code not in reasons:
            reasons.append(p.code)
    details = {"errors": [p.to_dict() for p in report.errors],
               "warnings": [p.to_dict() for p in report.warnings]}
    if report.ok or "meta_invalid" not in report.codes():
        try:
            meta = raw.meta
        except Exception:       # meta unreadable: the validator has the reason
            meta = None
        if meta is not None:
            r, d = rig_consistency(meta, rig, task_rig)
            reasons += [x for x in r if x not in reasons]
            details.update(d)
    return CheckResult(reasons, details)


class CheckStage(Stage):
    """**check**. Reads the whole ``raw/`` episode (manifest hashes, CSVs, meta.json, both MP4s decoded) and the
    rig yaml camera resolutions / fps. Writes ``processed/check.json`` (reasons, errors, warnings). Rejects on any
    contract error or C1 mismatch. Skipped when its inputs (MANIFEST.sha256 bytes, rig id, camera resolution / fps,
    task.yaml rig) are unchanged since a done / rejected / failed run."""

    name = "check"

    def inputs(self, ctx, ep) -> dict:
        s = ctx.sections
        return {"rig_id": s["rig_id"], "task_rig": ctx.task.rig, "cameras": s["cameras.resolution_fps"]}

    def run(self, ctx, ep) -> dict:
        res = check_episode(ep.raw, ctx.rig, ctx.task.rig)
        write_json(ep.processed / "check.json", {"reasons": res.reasons, **res.details})
        if not res.ok:
            raise StageFailed("rejected", res.reasons, details={"file": "check.json"})
        return {"file": "check.json", "n_warnings": len(res.details.get("warnings", []))}
