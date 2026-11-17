"""``umiproc status [TASK_FOLDER]``: what is in the data root, in plain words. Read only: it never creates, copies or
changes anything (docs/DESIGN.md "Where it runs and the data root").

* Without an argument: the data root and where it comes from; the rig file, SLAM config and lens file the
  processing reads (``rigs/``) and whether they match the repo; the umiproc_slam program and its vocabulary; the
  recordings taken so far by kind (``pi_sources.csv``); leftovers in ``incoming/``; one line per task folder.
* With a task folder (its name, e.g. ``block_raise_2``, or its path): its map, its gripper calibration, its
  episodes, the recordings waiting in ``.staging/`` and why, and the ``failed/`` ones with their reasons.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

__all__ = ["root_status", "folder_status", "format_root_status", "format_folder_status", "add_arguments", "run"]


# --------------------------------------------------------------------------------------------- gathering

def _root_source(arg) -> str:
    from umiproc.paths import ENV_ROOT

    if arg:
        return "--root"
    return f"${ENV_ROOT}" if os.environ.get(ENV_ROOT) else "the default"


def _rig_section(root: Path, rig_id: str) -> dict:
    from umiproc.sensors import RigFilesError, SensorError, check_rig_files, load_sensors, rig_copy_rules

    files = check_rig_files(root, rig_id)
    out = {"rig_id": rig_id, "rig": files.rig, "slam": files.slam, "lens_path": files.lens, "lens": None,
           "problem": None}
    if files.rig.state == "missing" or files.slam.state == "missing":
        if files.lens is not None:
            from umiproc.lens.lensfile import LensFileError, load_lens

            try:
                out["lens"] = load_lens(files.lens)
            except LensFileError as e:
                out["problem"] = str(e)
        return out
    try:
        rig_copy_rules(files.rig.path, files.rig.repo)        # read only: raises for a copy the processing refuses
        s = load_sensors(root, rig_id, ensure=False)
        out["lens"] = s.lens
    except (RigFilesError, SensorError) as e:
        out["problem"] = str(e)
    return out


def _slam_program() -> dict:
    from umiproc.slam.runner import FORK_DIR_ENV, default_binary, default_vocabulary

    b, v = default_binary(), default_vocabulary()
    return {"binary": b, "binary_ok": b.is_file() and os.access(b, os.X_OK), "vocabulary": v,
            "vocabulary_ok": v.is_file(), "env": FORK_DIR_ENV, "env_set": bool(os.environ.get(FORK_DIR_ENV))}


def root_status(root: Path, rig_id: str) -> dict:
    """Everything :func:`format_root_status` prints, as a dict (read only)."""
    from umiproc.recordings import count_by_kind, ledger_rows, task_folders

    root = Path(root)
    st = {"root": root, "exists": root.is_dir(), "rig": _rig_section(root, rig_id), "slam": _slam_program(),
          "kinds": {}, "statuses": {}, "incoming": [], "quarantine": 0, "folders": []}
    if not st["exists"]:
        return st
    st["kinds"] = count_by_kind(root)
    for r in ledger_rows(root):
        st["statuses"][r.get("status") or "?"] = st["statuses"].get(r.get("status") or "?", 0) + 1
    inc = root / "incoming"
    if inc.is_dir():
        st["incoming"] = sorted(p.name for p in inc.iterdir() if not p.name.startswith("."))
    q = root / "quarantine"
    if q.is_dir():
        st["quarantine"] = sum(1 for p in q.iterdir() if not p.name.startswith("."))
    st["folders"] = [folder_status(root, d.name) for d in task_folders(root)]
    return st


def _staging_reason(entry: Path, has_gripper: bool) -> str:
    from umiproc.dp.files import load_status
    from umiproc.dp.run.staging import waiting_for_slam

    status = load_status(entry / "processed")
    if waiting_for_slam(status):
        return "waiting for SLAM (umiproc_slam was not available when it ran)"
    if not has_gripper:
        return "waiting for the gripper calibration (run `umiproc calibrate`, then `umiproc run` again)"
    return "not finished (a run was interrupted; the next `umiproc run` picks it up)"


def folder_status(root: Path, folder: str) -> dict:
    """One task folder (read only). ``problem`` is set when the folder is not a usable task folder."""
    from umiproc.dp.config import ConfigError, load_task_config
    from umiproc.dp.files import load_status
    from umiproc.dp.intake import IntakeError, read_ledger
    from umiproc.dp.paths import episode_dirs
    from umiproc.dp.report import episode_summary
    from umiproc.dp.run.staging import failed_names, format_reason, read_reason, staging_names
    from umiproc.dp.stages.calibration import GRIPPER_RANGE_FILE
    from umiproc.recordings import CALIBRATION_DIR, map_slots

    d = Path(root) / "tasks" / folder
    st = {"folder": folder, "dir": d, "problem": None, "label": None, "rig": None, "created": None, "maps": [],
          "gripper_range": False, "calibrations": [], "episodes": [], "staging": [], "failed": [],
          "report": (d / "report.json").is_file()}
    if not d.is_dir():
        st["problem"] = f"no task folder {d}"
        return st
    try:
        task = load_task_config(d / "task.yaml")
        st["label"], st["rig"], st["created"] = task.label, task.rig, task.created
    except ConfigError as e:
        st["problem"] = str(e)
    try:
        ledger = read_ledger(d / "episodes.csv")
    except IntakeError as e:
        st["problem"] = st["problem"] or str(e)
        ledger = []
    pi = {r["slot"]: r.get("pi_name") or "" for r in ledger}
    st["maps"] = [p.name for p in map_slots(d)]
    st["gripper_range"] = (d / GRIPPER_RANGE_FILE).is_file()
    cal = d / CALIBRATION_DIR
    if cal.is_dir():
        st["calibrations"] = sorted(p.name for p in cal.iterdir() if not p.name.startswith(".") and p.is_dir())
    for e in episode_dirs(d):
        if not (e / "raw").exists():
            continue
        s = episode_summary(load_status(e / "processed"))
        st["episodes"].append({"slot": e.name, "pi_name": pi.get(e.name, ""), "status": s["status"],
                               "n_steps": s["n_steps"], "reasons": s["reasons"]})
    for name in staging_names(d):
        st["staging"].append({"pi_name": name, "why": _staging_reason(d / ".staging" / name, st["gripper_range"])})
    for name in failed_names(d):
        st["failed"].append({"pi_name": name, "reason": format_reason(read_reason(d, name))})
    return st


# --------------------------------------------------------------------------------------------- printing

def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _folder_line(f: dict) -> str:
    if f["problem"]:
        return f"  {f['folder']}: not a usable task folder ({f['problem']})"
    parts = [f"label {f['label']}", "map " + ("yes" if f["maps"] else "no"),
             "gripper calibration " + ("yes" if f["gripper_range"] else "no"), _plural(len(f["episodes"]), "episode")]
    if f["staging"]:
        parts.append(f"{len(f['staging'])} waiting")
    if f["failed"]:
        parts.append(f"{len(f['failed'])} failed")
    return f"  {f['folder']}: " + ", ".join(parts)


def format_root_status(st: dict, source: str) -> str:
    from umiproc.recordings import KINDS

    lines = [f"data root: {st['root']} (from {source})"]
    if not st["exists"]:
        lines.append("  it does not exist yet: the first umiproc command that takes a recording creates it")
    rs = st["rig"]
    lines += ["", f"rig {rs['rig_id']}: the files the processing reads (in {st['root'] / 'rigs'})"]
    for c in (rs["rig"], rs["slam"]):
        lines.append(f"  {c.what + ':':15s}{c.path.name}: {c.describe()}")
    if rs["lens_path"] is None:
        lines.append(f"  {'lens file:':15s}none, so SLAM uses the rig file's nominal lens model (\"nominal, "
                     "uncalibrated\"); optional: DIRECTIONS.md section 1")
    elif rs["lens"] is not None:
        src = rs["lens"].source or {}
        lines.append(f"  {'lens file:':15s}{rs['lens_path'].name}: calibrated {rs['lens'].date}"
                     + (f" from {src.get('pi_name') or src.get('episode_dir')}" if src else ""))
    else:
        lines.append(f"  {'lens file:':15s}{rs['lens_path'].name}: cannot be used")
    if rs["problem"]:
        lines.append(f"  PROBLEM: {rs['problem']}")
    sp = st["slam"]
    lines += ["", "SLAM program (built by umi-processing/native/build.sh):"]
    for what, path, ok in (("umiproc_slam", sp["binary"], sp["binary_ok"]),
                           ("vocabulary", sp["vocabulary"], sp["vocabulary_ok"])):
        lines.append(f"  {what + ':':15s}{path}: " + ("found" if ok else "NOT FOUND (build it with "
                                                       "umi-processing/native/build.sh)"))
    if sp["env_set"]:
        lines.append(f"  (${sp['env']} is set and chooses where these are)")
    if not st["exists"]:
        return "\n".join(lines)
    k = st["kinds"]
    taken = sum(k.values())
    lines += ["", "recordings taken (pi_sources.csv): " + (", ".join(f"{k.get(x, 0)} {x}" for x in KINDS)
                                                        + "".join(f", {v} {x}" for x, v in k.items()
                                                                  if x not in KINDS)
                                                        if taken else "none yet")]
    if taken and set(st["statuses"]) - {"ingested"}:
        lines.append("  by status: " + ", ".join(f"{v} {s}" for s, v in sorted(st["statuses"].items())))
    if st["quarantine"]:
        lines.append(f"  quarantine/: {_plural(st['quarantine'], 'recording')} intake refused (each has a "
                     "reason.json)")
    if st["incoming"]:
        lines.append(f"  incoming/ is not empty ({', '.join(st['incoming'][:5])}): an interrupted command left it; "
                     "the next command stops until it is moved away")
    lines += ["", "task folders (tasks/):" if st["folders"] else "task folders (tasks/): none yet (`umiproc map` "
                                                                    "creates one per session)"]
    lines += [_folder_line(f) for f in st["folders"]]
    return "\n".join(lines)


def format_folder_status(f: dict) -> str:
    from umiproc.dp.report import format_table

    if f["problem"] and f["label"] is None:
        return f"task folder {f['folder']}: {f['problem']}"
    lines = [f"task folder {f['dir']} (label {f['label']}, rig {f['rig']}, created {f['created']})"]
    if f["problem"]:
        lines.append(f"  PROBLEM: {f['problem']}")
    lines.append("  map:                  " + (f"yes ({', '.join('map/' + m for m in f['maps'])})" if f["maps"]
                                              else "no (`umiproc map` makes it when it creates the folder)"))
    if f["gripper_range"]:
        cal = f" (from calibration/{f['calibrations'][-1]})" if f["calibrations"] else ""
        lines.append(f"  gripper calibration:  yes, gripper_range.json{cal}")
    else:
        lines.append("  gripper calibration:  no (record one and run `umiproc calibrate`); demos wait for it")
    lines.append(f"  episodes:             {len(f['episodes'])}")
    if f["episodes"]:
        rows = [[e["slot"], e["pi_name"] or "-", e["status"], "-" if e["n_steps"] is None else e["n_steps"],
                 "; ".join(e["reasons"]) or "-"] for e in f["episodes"]]
        lines += ["    " + ln for ln in format_table(rows, ["slot", "recording", "status", "frames",
                                                              "problems"]).splitlines()]
    lines.append(f"  waiting in .staging/: {len(f['staging'])}")
    lines += [f"    {s['pi_name']}: {s['why']}" for s in f["staging"]]
    lines.append(f"  failed/:              {len(f['failed'])}")
    lines += [f"    {x['pi_name']}: {x['reason']}" for x in f["failed"]]
    if f["report"]:
        lines.append("  the last run's outcome per recording: report.json")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------- command

def add_arguments(p: argparse.ArgumentParser) -> None:
    from umiproc.sensors import DEFAULT_RIG

    p.add_argument("folder", nargs="?", metavar="TASK_FOLDER",
                   help="a task folder (its name under tasks/, e.g. block_raise_2, or its path); without it: the "
                        "whole data root")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--rig", default=DEFAULT_RIG, help=f"rig id whose files to show (default {DEFAULT_RIG})")


def run(args) -> int:
    from umiproc.dp.paths import DataRoot, resolve_task
    from umiproc.paths import data_root

    try:
        root = data_root(args.root)
    except ValueError as e:
        print(f"umiproc status: {e}", file=sys.stderr)
        return 1
    if args.folder is None:
        print(format_root_status(root_status(root, args.rig), _root_source(args.root)))
        return 0
    try:
        ref = resolve_task(args.folder, DataRoot(root))
    except ValueError as e:
        print(f"umiproc status: {e}", file=sys.stderr)
        return 1
    f = folder_status(root, ref.folder)
    print(format_folder_status(f))
    return 1 if f["label"] is None else 0
