"""``dataproc status``: per-episode processing state of a task folder, its ``.staging/`` and ``failed/``
recordings, and the calibration state. Read-only: never creates or changes anything (docs/PROCESS.md)."""

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
        self.exported: list[str] = []

    def run(self) -> dict:
        self.load_task()
        episodes = self.episode_rows()
        calibration = self.calibration_state()
        return {"task": self.folder, "label": self.task.label, "rig": self.task.rig, "episodes": episodes,
                "unfinished": self.unfinished_rows(), "calibration": calibration, "n_exported": len(self.exported)}

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
            if s["status"] == "exported" and (d / "processed" / "episode.zarr").is_dir():
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

    def unfinished_rows(self) -> list[dict]:
        """``.staging/<pi_name>`` (in progress, or waiting for SLAM: its last recorded stage) and
        ``failed/<pi_name>`` (``reason.json``: stage and reasons) entries of ``dataproc run``."""
        from dataproc.paths import FAILED_DIR, STAGING_DIR

        rows = []
        for d in _subdirs(self.task_dir / STAGING_DIR):
            status = load_status(d / "processed")
            s = episode_summary(status)
            waiting = "slam_unavailable" in ((status.get("slam") or {}).get("reasons") or [])
            rows.append({"entry": f"{STAGING_DIR}/{d.name}", "last_stage": s["last_stage"],
                         "state": "waiting for SLAM" if waiting else "in progress",
                         "reasons": ["slam: slam_unavailable"] if waiting else s["reasons"]})
        for d in _subdirs(self.task_dir / FAILED_DIR):
            rec = read_json(d / "reason.json", {})
            rec = rec if isinstance(rec, dict) else {}
            rows.append({"entry": f"{FAILED_DIR}/{d.name}", "state": "failed", "last_stage": rec.get("stage"),
                         "reasons": [str(r) for r in rec.get("reasons") or []] or [rec.get("message") or
                                                                                   "no reason.json"]})
        return rows


def _subdirs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def task_status(root: DataRoot, folder: str) -> dict:
    return TaskStatus(root, folder).run()


def format_status(st: dict) -> str:
    rows = [[e["slot"], e["pi_name"] or "-", e["last_stage"] or "-", e["status"],
             "-" if e["n_steps"] is None else e["n_steps"], "; ".join(e["reasons"]) or "-"]
            for e in st["episodes"]]
    c = st["calibration"]
    lines = [
        f"task {st['task']} (label {st['label']}, rig {st['rig']}): {len(st['episodes'])} episodes, "
        f"{st['n_exported']} exported",
        format_table(rows, ["slot", "pi_name", "last_stage", "status", "n_steps", "reasons"]),
    ]
    if st.get("unfinished"):
        lines += ["not episodes (never packaged):",
                  format_table([[u["entry"], u["last_stage"] or "-", u["state"], "; ".join(u["reasons"]) or "-"]
                                for u in st["unfinished"]], ["entry", "last_stage", "state", "reasons"])]
    lines.append(f"calibration: {c['state']}" + (f" ({c['name']})" if c.get("name") else "")
                 + (f": {c['reason']}" if c.get("reason") else ""))
    return "\n".join(lines)


def cli_status(root: DataRoot, folder: str) -> int:
    try:
        st = task_status(root, folder)
    except Exception as e:  # noqa: BLE001
        print(f"dataproc status: {e}", file=sys.stderr)
        return 1
    print(format_status(st))
    return 0
