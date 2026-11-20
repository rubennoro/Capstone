"""``teleop_train package TASK DATASET``: selection, consistency, the zip, package.json (PLAN.md §4.1, §6).
Owner: E5 (wave 2).

Flow (:func:`package_dataset`):

1. Refuse a bad or existing dataset name (unless ``overwrite``); then ``convert.convert_task`` for whatever is
   missing (or, without ``convert_missing``, only the marker check and the status.json files as they are).
2. Selection: the episodes whose ``status.json`` says ``converted`` and whose ``episode.zarr`` exists, in episode
   number order; ``fail`` (and unset) outcomes left out unless ``include_fail``; then the first ``episodes_n``.
3. Refuse a selected status of another ``CONVERT_VERSION`` / ``pose_source`` than asked, a mix of rig / TCP / pose
   source / convert version across the selection, an ``F_T_TCP`` other than the deployment's
   (``deployment/contracts/frames.yaml``, within 1e-9 per element) and an ``episode.zarr`` that does not match its
   status.
4. Build ``<dataset>.zarr.zip`` (``build_task_zarr_zip``) and ``package.json`` in a temporary sibling folder of the
   dataset folder and rename it into place, so a failure never leaves a half dataset (an overwritten dataset is
   replaced only once the new one is complete).

``package.json`` holds what ``rnoro/training``'s ``train/`` scripts read (``label_dir``, ``zip.sha256``,
``totals.n_steps``, ``totals.episodes``) in the layout of a handheld dataset, plus what makes the dataset traceable.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
import zarr

from teleop_train import __version__
from teleop_train import convert, paths
from teleop_train._copied.zarr_writer import build_task_zarr_zip
from teleop_train.paths import PathsError, TaskPaths

#: the TCP the deployment runs (PLAN §6 step 2, §8): this branch's copy of fd:contracts/frames.yaml, read as YAML
DEPLOY_FRAMES_YAML = Path(__file__).resolve().parents[2] / "deployment" / "contracts" / "frames.yaml"
#: the rig the deployment preps images with (fd policy_deploy config/frames_process.yaml rig_file, ea9f7fb)
DEPLOY_RIG_YAML = Path(__file__).resolve().parents[2] / "deployment" / "contracts" / "rigs" / "rig_pi_v3_640.yaml"
#: per element, as deploy_contracts/hello_check.py F_T_TCP_ABS_TOL
F_T_TCP_ABS_TOL = 1e-9
#: prompting's ``task/umi.yaml`` ``max_sequence_length`` (1200 steps) at 20 Hz (PLAN §7)
MAX_SEQUENCE_S = 60.0
#: ``check_name`` in ``rnoro/training``'s ``train/lib.sh`` (labels and dataset names)
TRAIN_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"
#: the status.json keys a selected episode must have
_STATUS_KEYS = ("state", "outcome", "n_steps", "kept_s", "pose_source", "convert_version", "rig_id", "rig_sha256",
                "F_T_TCP", "settings", "code_sha256")
#: compared across the selection (PLAN §6 step 2), in this order
_CONSISTENT = (("rig_id", "rig"), ("rig_sha256", "rig content (sha256)"), ("F_T_TCP", "F_T_TCP"),
               ("pose_source", "pose_source"), ("convert_version", "convert_version"))
_TRAIN_NAME = re.compile(TRAIN_NAME_RE)


class PackageError(ValueError):
    """A refusal; the message is one line saying what failed and what to check. Nothing was written."""


@dataclass
class PackageResult:
    dataset: str
    zip_path: Path
    package_json: Path
    record: dict
    warnings: list[str]
    convert_results: list = field(default_factory=list)    # convert.convert_task's results (convert_missing only)


@dataclass
class _Chosen:
    episode: str
    status: dict


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("task", metavar="TASK", help="the task folder under the raw root")
    p.add_argument("dataset", metavar="DATASET", help="a new dataset name, e.g. pick_v1 (letters, digits, '_', '-')")
    p.add_argument("--episodes", type=int, default=None, metavar="N",
                   help="only the first N usable episodes, in episode order (default: all)")
    p.add_argument("--include-fail", action="store_true", help="also package episodes saved as fail")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset of that name")


# --------------------------------------------------------------------------------------------- package_dataset

def package_dataset(tp: TaskPaths, dataset: str, pose_source: str, episodes_n: int | None = None,
                    include_fail: bool = False, overwrite: bool = False, convert_missing: bool = True,
                    progress: Callable | None = None) -> PackageResult:
    """Write ``<out_task_dir>/package/<dataset>/{<dataset>.zarr.zip, package.json}`` (PLAN §6). PackageError (one
    line, nothing written) on a bad or existing name, a refused selection or a failed build."""
    _check_names(tp.task, dataset)
    if pose_source not in paths.POSE_SOURCES:
        raise PackageError(f"pose source {pose_source!r}: use one of {', '.join(paths.POSE_SOURCES)}")
    if episodes_n is not None and episodes_n < 1:
        raise PackageError(f"--episodes must be at least 1 (got {episodes_n})")
    final = tp.package_dir(dataset)
    if (final.exists() or final.is_symlink()) and not overwrite:
        raise PackageError(f"dataset {final} already exists; choose another name or pass --overwrite")

    convert_results: list = []
    if convert_missing:
        try:
            convert_results = list(convert.convert_task(tp, pose_source, progress=progress))
        except ValueError as e:                     # PathsError (handheld folder, other raw folder) and the like
            raise PackageError(_one_line(e)) from None
    else:
        if not tp.out_task_dir.is_dir():
            raise PackageError(f"{tp.out_task_dir}: nothing converted yet; run teleop_train convert {tp.task}")
        try:
            paths.read_marker(tp)
        except PathsError as e:
            raise PackageError(str(e)) from None

    chosen, excluded = _select(tp, pose_source, episodes_n, include_fail)
    _check_consistent(chosen)
    _check_deploy_tcp(chosen)
    _check_deploy_rig(chosen)
    warnings = _warnings(chosen)
    record, zip_path = _build(tp, dataset, chosen, excluded, warnings, overwrite)
    return PackageResult(dataset=dataset, zip_path=zip_path, package_json=tp.package_json(dataset), record=record,
                         warnings=warnings, convert_results=convert_results)


def _check_names(task: str, dataset: str) -> None:
    try:
        paths.check_dataset_name(dataset)
    except PathsError as e:
        raise PackageError(str(e)) from None
    if not _TRAIN_NAME.fullmatch(dataset):
        raise PackageError(f"dataset name {dataset!r}: must start with a letter or digit (train/lib.sh check_name)")
    if not _TRAIN_NAME.fullmatch(task):
        raise PackageError(f"task {task!r}: train/ needs a label that starts with a letter or digit "
                           f"(train/lib.sh check_name); record under another task name")


def _one_line(e: BaseException) -> str:
    text = str(e).strip()
    return text.splitlines()[0] if text else type(e).__name__


# --------------------------------------------------------------------------------------------- selection

def _excluded(tp: TaskPaths, episode: str, state: str, reason: str) -> dict:
    return {"folder": tp.task, "entry": episode, "state": state, "reason": reason}


def _select(tp: TaskPaths, pose_source: str, episodes_n: int | None,
            include_fail: bool) -> tuple[list[_Chosen], list[dict]]:
    """The usable episodes in episode order, and the excluded entries (also in episode order)."""
    try:
        names = paths.raw_episodes(tp)
    except PathsError as e:
        raise PackageError(str(e)) from None
    chosen: list[_Chosen] = []
    excluded: list[dict] = []
    for ep in names:
        status_path = tp.status_json(ep)
        status = convert.read_status(status_path)
        if status is not None and status.get("schema") != convert.STATUS_SCHEMA:
            status = None
        if status is None:
            why = ("no status.json (not converted)" if not status_path.exists()
                   else "status.json unreadable or of another schema")
            excluded.append(_excluded(tp, ep, "excluded", f"{why}; run teleop_train convert {tp.task}"))
            continue
        if status.get("state") != "converted":
            excluded.append(_excluded(tp, ep, "excluded", str(status.get("reason") or "excluded by convert")))
            continue
        if not tp.episode_zarr(ep).is_dir():
            excluded.append(_excluded(tp, ep, "excluded", f"episode.zarr missing; run teleop_train convert "
                                                          f"{tp.task} --episodes {paths.episode_number(ep)} --force"))
            continue
        outcome = status.get("outcome")
        if outcome != "success" and not include_fail:
            what = f"outcome {outcome}" if outcome else "outcome not set"
            excluded.append(_excluded(tp, ep, "fail", f"{what} (use --include-fail)"))
            continue
        chosen.append(_Chosen(ep, status))

    if not chosen:
        counts: dict[str, int] = {}
        for x in excluded:
            counts[x["state"]] = counts.get(x["state"], 0) + 1
        parts = ", ".join(f"{n} {s}" for s, n in counts.items()) or "no episode folders"
        hint = "; --include-fail adds the fail ones" if counts.get("fail") else ""
        raise PackageError(f"nothing to package in {tp.out_task_dir}: 0 of {len(names)} episodes usable "
                           f"({parts}){hint}")
    if episodes_n is not None:
        if episodes_n > len(chosen):
            raise PackageError(f"--episodes {episodes_n}: only {len(chosen)} usable episodes in {tp.task}")
        for c in chosen[episodes_n:]:
            excluded.append(_excluded(tp, c.episode, "left out", f"--episodes {episodes_n} (only the first "
                                                                 f"{episodes_n})"))
        del chosen[episodes_n:]
        excluded.sort(key=lambda x: paths.episode_number(x["entry"]))

    want_version = convert.CONVERT_VERSION
    for c in chosen:
        missing = [k for k in _STATUS_KEYS if k not in c.status]
        if missing:
            raise PackageError(f"{tp.status_json(c.episode)}: missing {', '.join(missing)}; run teleop_train "
                               f"convert {tp.task} --force")
        if c.status["convert_version"] != want_version:
            raise PackageError(f"{c.episode}: converted by convert version {c.status['convert_version']}, this is "
                               f"{want_version}; run teleop_train convert {tp.task}")
        if c.status["settings"] != convert.SETTINGS or c.status["code_sha256"] != convert.code_sha256():
            raise PackageError(f"{c.episode}: converted with other settings or another teleop_train version; run "
                               f"teleop_train convert {tp.task}")
        if c.status["pose_source"] != pose_source:
            raise PackageError(f"{c.episode}: converted with pose source {c.status['pose_source']}, not "
                               f"{pose_source}; run teleop_train convert {tp.task} --pose-source {pose_source}")
        _check_zarr(tp, c)
    return chosen, excluded


def _check_zarr(tp: TaskPaths, c: _Chosen) -> None:
    """The episode.zarr is the one its status.json describes: name, task label, step count."""
    path = tp.episode_zarr(c.episode)
    redo = f"run teleop_train convert {tp.task} --episodes {paths.episode_number(c.episode)} --force"
    try:
        root = zarr.open_group(store=zarr.DirectoryStore(str(path)), mode="r")
        name, label = root.attrs.get("name"), root.attrs.get("task_label")
        n = int(root["main_rgb"].shape[0])
    except Exception as e:  # noqa: BLE001 - any unreadable zarr is a refusal
        raise PackageError(f"{path}: unreadable ({type(e).__name__}); {redo}") from None
    want_name = f"{tp.task}_{c.episode}"
    if name != want_name or label != tp.task:
        raise PackageError(f"{path}: name {name!r} / task label {label!r}, expected {want_name!r} / {tp.task!r}; "
                           f"{redo}")
    if n != c.status["n_steps"]:
        raise PackageError(f"{path}: {n} steps but its status.json says {c.status['n_steps']}; {redo}")


def _check_consistent(chosen: list[_Chosen]) -> None:
    """One rig, one TCP, one pose source, one convert version across the selection (PLAN §6 step 2)."""
    first = chosen[0]
    for c in chosen[1:]:
        for key, what in _CONSISTENT:
            if c.status[key] != first.status[key]:
                raise PackageError(f"{c.episode} has another {what} than {first.episode} ({c.status[key]} vs "
                                   f"{first.status[key]}); a dataset deploys with one rig and one TCP: package "
                                   f"them separately")


def deploy_F_T_TCP(path: Path | None = None) -> np.ndarray:
    """F_T_TCP (4x4 float64) of the deployment's frames.yaml (default DEPLOY_FRAMES_YAML). PackageError if unreadable."""
    path = Path(path) if path is not None else DEPLOY_FRAMES_YAML
    try:
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))["F_T_TCP"]
        m = np.array(rows, dtype=np.float64)
        if m.shape != (4, 4) or not np.all(np.isfinite(m)):
            raise ValueError("not 4x4 finite")
    except (OSError, UnicodeDecodeError, yaml.YAMLError, KeyError, TypeError, ValueError) as e:
        raise PackageError(f"{path}: cannot read F_T_TCP ({type(e).__name__}); the deployment's TCP is needed to "
                           f"check the dataset (PLAN §6)") from None
    return m


def _check_deploy_tcp(chosen: list[_Chosen]) -> None:
    """The shared F_T_TCP is the deployment's (within F_T_TCP_ABS_TOL per element, as the hello check)."""
    want = deploy_F_T_TCP()
    try:
        got = np.array(chosen[0].status["F_T_TCP"], dtype=np.float64)
    except (TypeError, ValueError):
        got = None
    if got is None or got.shape != (4, 4):
        raise PackageError(f"{chosen[0].episode}: status.json F_T_TCP is not a 4x4 matrix; run teleop_train convert "
                           f"--force")
    diff = float(np.max(np.abs(got - want)))
    if not diff <= F_T_TCP_ABS_TOL:
        raise PackageError(f"the episodes' F_T_TCP differs from the deployment's {DEPLOY_FRAMES_YAML} (max "
                           f"|diff| {diff:.3g} > {F_T_TCP_ABS_TOL:g}); a policy deployed with another TCP moves in "
                           f"the wrong frame: record with the deployment's TCP (PLAN §8)")


def deploy_rig_sha256(path: Path | None = None) -> str:
    """convert.rig_sha256 of the deployment's rig file (default DEPLOY_RIG_YAML). PackageError if unreadable."""
    path = Path(path) if path is not None else DEPLOY_RIG_YAML
    try:
        rig = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(rig, dict):
            raise ValueError("not a mapping")
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as e:
        raise PackageError(f"{path}: cannot read the deployment rig ({type(e).__name__}); it is needed to check the "
                           f"dataset (PLAN §6)") from None
    return convert.rig_sha256(rig)


def _check_deploy_rig(chosen: list[_Chosen]) -> None:
    """The shared recorded rig is the deployment's: the ultrawide prep depends on its lens model (PLAN §8)."""
    want = deploy_rig_sha256()
    if chosen[0].status["rig_sha256"] != want:
        raise PackageError(f"the episodes were recorded with rig {chosen[0].status['rig_id']} (sha256 "
                           f"{str(chosen[0].status['rig_sha256'])[:12]}), not the deployment's {DEPLOY_RIG_YAML.name} "
                           f"({want[:12]}): their ultrawide images would differ from what the policy sees live; "
                           f"record with the deployment rig (PLAN §8)")


def _warnings(chosen: list[_Chosen]) -> list[str]:
    out = []
    for c in chosen:
        kept_s = float(c.status["kept_s"])
        if kept_s > MAX_SEQUENCE_S:
            out.append(f"{c.episode}: {kept_s:.1f} s after trimming, over the {MAX_SEQUENCE_S:.0f} s of "
                       f"max_sequence_length 1200 at 20 Hz; for prompting add "
                       f"task.shape_meta.max_sequence_length=<steps> (PLAN §7)")
    return out


# --------------------------------------------------------------------------------------------- build

def _build(tp: TaskPaths, dataset: str, chosen: list[_Chosen], excluded: list[dict], warnings: list[str],
           overwrite: bool) -> tuple[dict, Path]:
    final = tp.package_dir(dataset)
    parent = final.parent
    created_parent = not parent.exists()
    try:
        parent.mkdir(exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{dataset}.tmp-", dir=parent))
    except OSError as e:
        raise PackageError(f"{parent}: cannot create ({type(e).__name__}: {e}); check --out-root") from None
    zip_name = tp.zip_path(dataset).name
    try:
        _chmod_default(tmp, 0o777)
        tmp_zip = tmp / zip_name
        try:
            summary = build_task_zarr_zip([tp.episode_zarr(c.episode) for c in chosen], tmp_zip, tmp_dir=tmp)
        except Exception as e:  # noqa: BLE001 - reported as a refusal; nothing is left behind
            raise PackageError(f"building {zip_name} failed ({type(e).__name__}: {_one_line(e)})") from None
        _chmod_default(tmp_zip, 0o666)
        n_steps = sum(int(c.status["n_steps"]) for c in chosen)
        if summary["n_steps"] != n_steps:
            raise PackageError(f"{zip_name}: {summary['n_steps']} steps but the status files say {n_steps}; "
                               f"run teleop_train convert {tp.task} --force")
        record = _record(tp, dataset, chosen, excluded, warnings, tmp_zip)
        with open(tmp / paths.PACKAGE_JSON, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _install(tmp, final, overwrite)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        if created_parent:
            try:
                parent.rmdir()
            except OSError:
                pass
        raise
    return record, final / zip_name


def _chmod_default(path: Path, mode: int) -> None:
    """mkdtemp / mkstemp make 0700 / 0600; give the dataset the usual permissions (``mode`` minus the umask)."""
    um = os.umask(0)
    os.umask(um)
    os.chmod(path, mode & ~um)


def _install(tmp: Path, final: Path, overwrite: bool) -> None:
    """Rename the finished ``tmp`` onto ``final``. An existing ``final`` (``overwrite`` only) is moved aside first
    and removed after the rename; if the rename fails, it is put back."""
    old = None
    if final.exists() or final.is_symlink():
        if not overwrite:
            raise PackageError(f"dataset {final} appeared while packaging; choose another name or pass --overwrite")
        old = Path(tempfile.mkdtemp(prefix=f".{final.name}.old-", dir=final.parent))
        os.rmdir(old)                               # a unique free name to move the old dataset onto
        os.rename(final, old)
    try:
        os.rename(tmp, final)
    except BaseException:
        if old is not None and not final.exists():
            os.rename(old, final)
        raise
    if old is not None:
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _record(tp: TaskPaths, dataset: str, chosen: list[_Chosen], excluded: list[dict], warnings: list[str],
            zip_path: Path) -> dict[str, Any]:
    first = chosen[0].status
    task = tp.task
    return {
        "dataset": dataset,
        "label_dir": task,
        "labels": [task],
        "folders": [{"folder": task, "label": task, "rig": first["rig_id"]}],
        "rig": first["rig_id"],
        "episodes": [{"folder": task, "slot": c.episode, "pi_name": f"{task}_{c.episode}", "label": task,
                      "n_steps": int(c.status["n_steps"]), "kept_s": c.status["kept_s"],
                      "outcome": c.status["outcome"]} for c in chosen],
        "excluded": excluded,
        "totals": {"episodes": len(chosen), "n_steps": sum(int(c.status["n_steps"]) for c in chosen),
                   "per_label": {task: len(chosen)}, "excluded": len(excluded)},
        "zip": {"name": zip_path.name, "size_bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path)},
        "source": {"kind": paths.MARKER_SOURCE, "raw_task_dir": str(tp.raw_task_dir)},
        "pose_source": first["pose_source"],
        "F_T_TCP": first["F_T_TCP"],
        "warnings": list(warnings),
        "teleop_train_version": __version__,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------------------------- command line

def next_commands(task: str, dataset: str) -> list[str]:
    """The ``train/all.sh`` lines to run next (PLAN §7): plain diffusion, then behavior prompting."""
    return [f"cd ~/Capstone && train/all.sh {task} {dataset} --config umi_policy_dunet",
            f"cd ~/Capstone && train/all.sh {task} {dataset} --config umi_policy_dunetp"]


def format_summary(res: PackageResult, out_root: Path) -> str:
    rec = res.record
    lines: list[str] = []
    if res.convert_results:
        counts: dict[str, int] = {}
        for r in res.convert_results:
            state = getattr(r, "state", "?")
            counts[state] = counts.get(state, 0) + 1
        lines.append("convert: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    lines.append(f"packaged {rec['totals']['episodes']} episodes of {rec['label_dir']} "
                 f"(pose source {rec['pose_source']}, rig {rec['rig']}):")
    for e in rec["episodes"]:
        lines.append(f"  {e['slot']}  {str(e['outcome']):8s} {float(e['kept_s']):7.1f} s {e['n_steps']:7d} steps")
    if rec["excluded"]:
        lines.append(f"left out ({len(rec['excluded'])}):")
        for x in rec["excluded"]:
            lines.append(f"  {x['entry']}  {x['state']:9s} {x['reason']}")
    lines += [f"warning: {w}" for w in res.warnings]
    lines += [f"wrote {res.zip_path} ({rec['zip']['size_bytes'] / 1e6:.1f} MB, {rec['totals']['episodes']} "
              f"episodes, {rec['totals']['n_steps']} steps)",
              f"      {res.package_json}", ""]
    plain, prompting = next_commands(rec["label_dir"], rec["dataset"])
    lines += ["next, plain diffusion:", f"  {plain}", "or behavior prompting:", f"  {prompting}"]
    if out_root != paths.DEFAULT_OUT_ROOT.expanduser().resolve():
        lines.append(f"(--out-root is not {paths.DEFAULT_OUT_ROOT}: LOCAL_DATA_ROOT in train/config.local.env "
                     f"must point at {out_root})")
    return "\n".join(lines)


def _print_convert_row(result: Any) -> None:
    """convert_task's progress: one row per episode as it is checked (convert's own row format)."""
    print(f"convert {convert.format_row(result)}", flush=True)


def run(args: argparse.Namespace) -> int:
    """0 written; 1 nothing written (one-line reason on stderr)."""
    try:
        tp = paths.task_paths(args.task, args.raw_root, args.out_root)
        res = package_dataset(tp, args.dataset, args.pose_source, episodes_n=args.episodes,
                              include_fail=args.include_fail, overwrite=args.overwrite,
                              progress=_print_convert_row)
    except (PackageError, PathsError) as e:
        print(f"teleop_train package: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return 1
    print(format_summary(res, Path(args.out_root).expanduser().resolve()))
    return 0
