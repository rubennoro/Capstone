"""``dataproc status --task NAME``: per-episode processing state, calibration state, and whether the output zip is
up to date. Read-only: never creates or changes anything (docs/PROCESS.md)."""

from __future__ import annotations

import sys
from pathlib import Path

from dataproc.files import load_status, read_json
from dataproc.paths import DataRoot, episode_dirs
from dataproc.report import episode_summary, format_table

__all__ = ["TaskStatus", "task_status", "format_status", "cli_status"]


class TaskStatus:
    """Collects what ``dataproc status`` prints::

        st = TaskStatus(root, folder).run()     # dict; FileNotFoundError if the folder / task.yaml is missing
    """

    def __init__(self, root: DataRoot, folder: str):
        self.root, self.folder = root, folder
        self.task_dir = root.tasks / folder
        self.task = None
        self.ledger: list[dict] = []
        self.newest_zarr_mtime: float | None = None
        self.exported: list[str] = []

    def run(self) -> dict:
        self.load_task()
        episodes = self.episode_rows()
        calibration = self.calibration_state()
        output = self.output_state()
        return {"task": self.folder, "label": self.task.label, "rig": self.task.rig, "episodes": episodes,
                "calibration": calibration, "n_exported": len(self.exported), **output}

    def load_task(self) -> None:
        """Reads ``task.yaml`` and ``episodes.csv``."""
        from dataproc.config import load_task_config
        from dataproc.intake import read_ledger

        if not self.task_dir.is_dir():
            raise FileNotFoundError(f"no task folder {self.task_dir} (run intake first)")
        if not (self.task_dir / "task.yaml").is_file():
            raise FileNotFoundError(f"{self.task_dir / 'task.yaml'} is missing (run intake first)")
        self.task = load_task_config(self.task_dir / "task.yaml")
        self.ledger = read_ledger(self.task_dir / "episodes.csv")

    def episode_rows(self) -> list[dict]:
        """One row per ``episode_N`` with a ``raw/`` (from its ``processed/status.json``; ``pending`` if none)."""
        pi = {r["slot"]: r.get("pi_name") or "" for r in self.ledger}
        rows = []
        for d in episode_dirs(self.task_dir):
            if not (d / "raw").is_dir():
                continue                   # an empty episode_N/ from an interrupted intake
            status = load_status(d / "processed")
            s = episode_summary(status)
            rows.append({"slot": d.name, "pi_name": pi.get(d.name, ""), **s})
            z = d / "processed" / "episode.zarr"
            if s["status"] == "exported" and z.is_dir():
                m = _mtime_tree(z)
                self.newest_zarr_mtime = m if self.newest_zarr_mtime is None else max(self.newest_zarr_mtime, m)
                self.exported.append(status["export"].get("outputs", {}).get("name") or pi.get(d.name, ""))
        return rows

    def calibration_state(self) -> dict:
        """The newest calibration's recorded result: ``done`` | ``calibration_failed`` | ``not_processed`` |
        ``no_calibration``."""
        from dataproc.stages.calibration import GRIPPER_RANGE_FILE, newest_calibration

        cal_dir = newest_calibration(self.task_dir, self.ledger)
        if cal_dir is None:
            out = {"state": "no_calibration", "name": None, "reason": None}
        else:
            rec = load_status(cal_dir / "processed").get("gripper_range")
            state = "not_processed" if rec is None else ("done" if rec.get("status") == "done"
                                                         else "calibration_failed")
            out = {"state": state, "name": cal_dir.name, "reason": rec.get("reason") if rec else None}
        out["task_gripper_range"] = (self.task_dir / GRIPPER_RANGE_FILE).is_file()
        return out

    def output_state(self) -> dict:
        """The zip is ``up to date`` when it exists, the last ``report.json`` says it was built (``zip`` not null),
        its ``meta/episode_names`` are exactly the names of the episodes exported now, in order, and it is newer
        than every exported ``episode.zarr``. Otherwise ``stale`` (a kept older zip) or ``missing``."""
        from dataproc.process import REPORT_FILE, zip_path

        zp = zip_path(self.task_dir, self.folder)
        if not zp.is_file():
            state = "missing" if self.exported else "missing (nothing exportable)"
            return {"zip": str(zp), "zip_exists": False, "zip_state": state}
        report = read_json(self.task_dir / "output" / REPORT_FILE)
        built = isinstance(report, dict) and report.get("zip") is not None
        names = _zip_episode_names(zp)
        stale = (not built or names != self.exported
                 or (self.newest_zarr_mtime is not None and self.newest_zarr_mtime > zp.stat().st_mtime))
        return {"zip": str(zp), "zip_exists": True, "zip_episode_names": names,
                "zip_state": "stale (run dataproc process)" if stale else "up to date"}


def _zip_episode_names(zp: Path) -> list[str] | None:
    """``meta/episode_names`` of a task zip (None if it cannot be read)."""
    import zarr

    try:
        store = zarr.ZipStore(str(zp), mode="r")
        try:
            return [str(x) for x in zarr.open_group(store, mode="r")["meta/episode_names"][:]]
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        return None


def _mtime_tree(p: Path) -> float:
    best = p.stat().st_mtime
    for q in p.rglob("*"):
        try:
            best = max(best, q.stat().st_mtime)
        except OSError:
            pass
    return best


def task_status(root: DataRoot, folder: str) -> dict:
    return TaskStatus(root, folder).run()


def format_status(st: dict) -> str:
    rows = [[e["slot"], e["pi_name"] or "-", e["last_stage"] or "-", e["status"],
             "-" if e["n_steps"] is None else e["n_steps"], "; ".join(e["reasons"]) or "-"]
            for e in st["episodes"]]
    c = st["calibration"]
    return "\n".join([
        f"task {st['task']} (label {st['label']}, rig {st['rig']}): {len(st['episodes'])} episodes, "
        f"{st['n_exported']} exported",
        format_table(rows, ["slot", "pi_name", "last_stage", "status", "n_steps", "reasons"]),
        f"calibration: {c['state']}" + (f" ({c['name']})" if c.get("name") else "")
        + (f": {c['reason']}" if c.get("reason") else ""),
        f"output: {st['zip']}: {st['zip_state']}",
    ])


def cli_status(root: DataRoot, folder: str) -> int:
    try:
        st = task_status(root, folder)
    except Exception as e:  # noqa: BLE001
        print(f"dataproc status: {e}", file=sys.stderr)
        return 1
    print(format_status(st))
    return 0
