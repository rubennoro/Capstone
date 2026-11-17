"""``umiproc status [TASK_FOLDER]``: what is in the data root, in plain words. Read only: it never creates, copies or
changes anything (docs/DESIGN.md "Where it runs and the data root").

* Without an argument: the data root and where it comes from; the rig file, SLAM config and lens file the
  processing reads (``rigs/``) and whether they match the repo; the umiproc_slam program and its vocabulary; the
  recordings taken so far by kind (``pi_sources.csv``); leftovers in ``incoming/``; one line per task folder.
* With a task folder (its name, e.g. ``block_raise_2``, or its path): its map, its gripper calibration, its
  episodes, the recordings waiting in ``.staging/`` and why, and the ``failed/`` ones with their reasons.

The map (:mod:`umiproc.map.stored`): whether it passed, when, its fingerprint's short hash, and whether the current
data-root files and SLAM build still give that fingerprint (else **out of date**: re-run the map, then every
episode). The root view also lists the failed maps kept in ``maps_failed/``.
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
          "kinds": {}, "statuses": {}, "incoming": [], "quarantine": 0, "folders": [], "failed_maps": []}
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
    cache: dict = {}
    st["folders"] = [folder_status(root, d.name, _fingerprints=cache) for d in task_folders(root)]
    from umiproc.map.stored import failed_maps

    st["failed_maps"] = failed_maps(root)
    return st


def _staging_reason(entry: Path, has_gripper: bool) -> str:
    """Why a recording is in ``.staging/``, from its ``processed/localize.json`` (umiproc run)."""
    from umiproc.dp.files import read_json
    from umiproc.run.recording import LOCALIZE_FILE, LOCALIZED

    loc = read_json(entry / "processed" / LOCALIZE_FILE, {}) or {}
    if loc.get("phase") == LOCALIZED:
        ok = [p for p in loc.get("pieces") or [] if (p.get("pose") or {}).get("ok")]
        head = (f"localized against map/{(loc.get('map') or {}).get('pi_name')}, {len(ok)} piece(s) passed every "
                "pose check")
        if not has_gripper:
            return head + ": waiting for the gripper calibration (run `umiproc calibrate`, then `umiproc run` again)"
        return head + ": the next `umiproc run` of this folder finishes it (widths, export)"
    if not has_gripper and not loc:
        return "not localized yet (an interrupted run; the next `umiproc run` picks it up)"
    return "not finished (a run was interrupted; the next `umiproc run` picks it up)"


def _calibration_info(d: Path, calibrations: list[str]) -> dict | None:
    """The newest calibration's ``umiproc_calibrate.json`` (umiproc calibrate), else None."""
    from umiproc.calibrate.gripper import REPORT_NAME
    from umiproc.dp.files import read_json

    for name in reversed(calibrations):
        rep = read_json(d / "calibration" / name / "processed" / REPORT_NAME)
        if isinstance(rep, dict):
            return {"name": name, **rep}
    return None


def _current_fingerprint(root: Path, rig_id: str | None, cache: dict) -> tuple[str | None, dict | None, str | None]:
    """``(hash, record, why not)`` of the fingerprint a map would get now (cached per rig)."""
    from umiproc.map.stored import current_fingerprint

    if rig_id not in cache:
        try:
            h, rec = current_fingerprint(root, rig_id)
            cache[rig_id] = (h, rec, None)
        except Exception as e:  # noqa: BLE001 - any reason the current one cannot be made is shown as such
            cache[rig_id] = (None, None, str(e))
    return cache[rig_id]


def _map_section(root: Path, d: Path, rig_id: str | None, cache: dict) -> list[dict]:
    from umiproc.map.stored import fingerprint_differences, read_map

    out = []
    for m in read_map(d):
        info = dict(m)
        info["up_to_date"], info["differences"], info["cannot_compare"] = None, [], None
        if m["complete"] and m["fingerprint_hash"]:
            h, rec, why = _current_fingerprint(root, rig_id, cache)
            if h is None:
                info["cannot_compare"] = why
            else:
                info["up_to_date"] = h == m["fingerprint_hash"]
                if not info["up_to_date"]:
                    info["differences"] = fingerprint_differences(m["record"], rec)
        out.append(info)
    return out


def folder_status(root: Path, folder: str, *, _fingerprints: dict | None = None) -> dict:
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
          "map_info": [], "calibration_info": None,
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
    st["map_info"] = _map_section(Path(root), d, st["rig"], {} if _fingerprints is None else _fingerprints)
    st["gripper_range"] = (d / GRIPPER_RANGE_FILE).is_file()
    cal = d / CALIBRATION_DIR
    if cal.is_dir():
        st["calibrations"] = sorted(p.name for p in cal.iterdir() if not p.name.startswith(".") and p.is_dir())
    st["calibration_info"] = _calibration_info(d, st["calibrations"]) if st["gripper_range"] else None
    from umiproc.dp.files import read_json
    from umiproc.run.recording import PIECE_FILE

    for e in episode_dirs(d):
        if not (e / "raw").exists():
            continue
        s = episode_summary(load_status(e / "processed"))
        piece = read_json(e / "processed" / PIECE_FILE, {}) or {}
        st["episodes"].append({"slot": e.name, "pi_name": pi.get(e.name, "") or piece.get("pi_name", ""),
                               "status": s["status"], "n_steps": s["n_steps"], "reasons": s["reasons"],
                               "piece": piece.get("piece"), "pieces": piece.get("pieces"),
                               "frames": piece.get("frames"), "raw_of": piece.get("raw_of")})
    for name in staging_names(d):
        st["staging"].append({"pi_name": name, "why": _staging_reason(d / ".staging" / name, st["gripper_range"])})
    for name in failed_names(d):
        st["failed"].append({"pi_name": name, "reason": format_reason(read_reason(d, name))})
    return st


# --------------------------------------------------------------------------------------------- printing

def _episode_groups(episodes: list[dict]) -> list[list[dict]]:
    """The episodes grouped by recording: consecutive slots with the same ``pi_name`` (the pieces of one split
    recording) form one group."""
    groups: list[list[dict]] = []
    for e in episodes:
        if groups and e["pi_name"] and groups[-1][0]["pi_name"] == e["pi_name"]:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _folder_line(f: dict) -> str:
    if f["problem"]:
        return f"  {f['folder']}: not a usable task folder ({f['problem']})"
    parts = [f"label {f['label']}", "map " + (_map_word(f["map_info"]) if f["maps"] else "no"),
             "gripper calibration " + ("yes" if f["gripper_range"] else "no"), _plural(len(f["episodes"]), "episode")]
    if f["staging"]:
        parts.append(f"{len(f['staging'])} waiting")
    if f["failed"]:
        parts.append(f"{len(f['failed'])} failed")
    return f"  {f['folder']}: " + ", ".join(parts)


def _map_word(infos: list[dict]) -> str:
    if not infos:
        return "no"
    m = infos[-1]
    if not m["complete"]:
        return "INCOMPLETE (no report.json: an interrupted map)"
    word = "passed" if m["passed"] else "FAILED"
    if m["up_to_date"] is False:
        word += ", OUT OF DATE"
    elif m["up_to_date"] is None and m.get("cannot_compare"):
        word += ", up to date: cannot tell"
    return word


def _map_lines(infos: list[dict]) -> list[str]:
    if not infos:
        return ["  map:                  no (`umiproc map` makes it when it creates the folder)"]
    out = []
    for m in infos:
        head = f"  map:                  map/{m['pi_name']}"
        if not m["complete"]:
            out.append(head + ": INCOMPLETE: the recording was taken but its map was not finished (an interrupted "
                       "`umiproc map`). The clip is in the ledger, so `umiproc map` on it again stops with \"already "
                       "taken\": record a new mapping clip (it creates a new session folder)")
            if m.get("work_report"):
                out.append(f"                        its map is still in {m['work_report'].parent}: it can be moved "
                           "into this folder by hand before the next `umiproc map` clears it (DIRECTIONS.md section "
                           "3, \"An interrupted map\")")
            continue
        checks = ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in m["checks"].items())
        out.append(head + f": {'passed' if m['passed'] else 'FAILED'} ({checks}), made {m['created_utc']}")
        b, v = m.get("build") or {}, m.get("vio_inflation") or {}
        out.append(f"                        fingerprint {str(m['fingerprint_hash'])[:12]}: SLAM build "
                   f"{b.get('binary')} (patches {b.get('patches_label')}), IMU vio_inflation "
                   f"{v.get('noise_density')}/{v.get('random_walk')}")
        if m["up_to_date"] is True:
            out.append("                        up to date (the current rig files and SLAM build give the same "
                       "fingerprint)")
        elif m["up_to_date"] is False:
            out.append("                        OUT OF DATE: the current settings or SLAM build differ in "
                       + ", ".join(m["differences"][:8]) + "; re-run the map (a new session folder), then every "
                       "episode")
        else:
            out.append(f"                        up to date: cannot tell ({m.get('cannot_compare')})")
    return out


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
    fm = st.get("failed_maps") or []
    if fm:
        lines += ["", "failed maps (maps_failed/, newest first; nothing was taken, each can be run again):"]
        for x in fm[:5]:
            why = x["error"] or (", ".join(x["failed"]) + " failed" if x["failed"] else "see report.json")
            lines.append(f"  {x['pi_name']} at {x['when']}: {why} ({x['dir']}/report.json)")
        if len(fm) > 5:
            lines.append(f"  ... and {len(fm) - 5} older")
    return "\n".join(lines)


def format_folder_status(f: dict) -> str:
    from umiproc.dp.report import format_table

    if f["problem"] and f["label"] is None:
        return f"task folder {f['folder']}: {f['problem']}"
    lines = [f"task folder {f['dir']} (label {f['label']}, rig {f['rig']}, created {f['created']})"]
    if f["problem"]:
        lines.append(f"  PROBLEM: {f['problem']}")
    lines += _map_lines(f["map_info"])
    if f["gripper_range"]:
        ci = f.get("calibration_info")
        cal = f" (from calibration/{f['calibrations'][-1]})" if f["calibrations"] else ""
        lines.append(f"  gripper calibration:  yes, gripper_range.json{cal}")
        if ci:
            op = ci.get("opening") or {}
            rule = ci.get("finger_tag_rule") or {}
            if op.get("measured_m") is not None:
                lines.append(f"                        opening {op['measured_m']:.3f} m (closed {op['min_width']:.3f} m"
                             f", open {op['max_width']:.3f} m); finger tags: {rule.get('message', '-')}")
            lines += [f"                        warning: {w}" for w in ci.get("warnings") or []]
    else:
        lines.append("  gripper calibration:  no (record one and run `umiproc calibrate`); demos wait for it")
    groups = _episode_groups(f["episodes"])
    lines.append(f"  episodes:             {len(f['episodes'])}"
                 + (f" (from {len(groups)} recordings)" if len(groups) != len(f["episodes"]) else ""))
    if f["episodes"]:
        rows = []
        for g in groups:
            e0 = g[0]
            slots = ", ".join(e["slot"] for e in g)
            n = [e["n_steps"] for e in g]
            if e0.get("pieces") and e0["pieces"] > 1:
                kept = ", ".join(str(e.get("piece")) for e in g)
                rec = f"{e0['pi_name'] or '-'} (piece{'s' if len(g) > 1 else ''} {kept} of {e0['pieces']})"
            else:
                rec = e0["pi_name"] or "-"
            st_ = ", ".join(dict.fromkeys(e["status"] for e in g))
            steps = "+".join("-" if x is None else str(x) for x in n)
            rows.append([slots, rec, st_, steps, "; ".join(r for e in g for r in e["reasons"]) or "-"])
        lines += ["    " + ln for ln in format_table(rows, ["slot", "recording", "status", "steps",
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
