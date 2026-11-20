# New. `policy_deploy use RUN_FOLDER | hold_still`: points station.yaml at a training run, so nobody edits it by hand.
"""UsePolicy: rewrites the `policy:` and `manager:` sections of station.yaml from a fetched training run.

A run folder is what train/fetch.sh leaves in data/<folder>/policy/<run>/: checkpoints/*.ckpt and
run_config/config.yaml. From it come the checkpoint, whether it is a prompting checkpoint (use_prompting) and, for a
prompting one, the dataset it trained on (data/<folder>/package/<dataset>/<dataset>.zarr.zip) and its task name.

Only those two sections change; every other line of station.yaml (comments included) stays as it is. Keys of the two
sections that are not about the policy choice (precision, diffusion_steps, exec_h, ...) are kept unless a flag sets
them. policy.mock_kind follows policy.kind, so `run --mock` runs the chosen model. The new file is checked with
Station.load (every key) before it replaces the old one; the speed flags are range-checked here. The policy section's
values are the policy process's to check: `policy_deploy check`, the next step (the launcher never imports a process
subpackage, tests/test_import_rules.py).
"""
from __future__ import annotations

import difflib
import json
import os
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from policy_deploy.contracts import ACTION_STEPS, paths
from policy_deploy.contracts.station import Station, StationError

HOLD_STILL = "hold_still"
SECTIONS = ("policy", "manager")
#: keys that belong to the policy choice: removed or replaced by every `use`
POLICY_CHOICE_KEYS = ("kind", "mock_kind", "checkpoint", "prompt_dataset", "prompt_split")
MANAGER_CHOICE_KEYS = ("task_name", "prompt_index")


class UseError(Exception):
    """One line saying what is wrong, and a hint."""

    def __init__(self, what: str, hint: str = "") -> None:
        self.what = what
        self.hint = hint
        super().__init__(what if not hint else f"{what} ({hint})")


@dataclass
class Choice:
    label: str                         # the run folder's name, or hold_still
    policy: dict[str, Any]             # the keys of POLICY_CHOICE_KEYS to set
    manager: dict[str, Any]            # the keys of MANAGER_CHOICE_KEYS to set
    notes: list[str]


# ------------------------------------------------------------------------------------------------ reading a run


def _home_path(p: Path) -> str:
    """`~/...` when under the home folder (station.yaml expands ~), else the absolute path."""
    p = p.expanduser().absolute()
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def find_checkpoint(run: Path) -> Path:
    ckpts = sorted((run / "checkpoints").glob("*.ckpt"))
    if not ckpts:
        raise UseError(f"no checkpoint in {run / 'checkpoints'}",
                       f"fetch it first: train/fetch.sh <folder> {run.name}")
    if len(ckpts) == 1:
        return ckpts[0]
    index = run / "checkpoints" / "checkpoints.json"           # several: the best one (rank 1)
    try:
        ranked = sorted(json.loads(index.read_text()), key=lambda e: e["rank"])
        best = run / "checkpoints" / ranked[0]["file"]
        if best.is_file():
            return best
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    raise UseError(f"{len(ckpts)} checkpoints in {run / 'checkpoints'} and no usable checkpoints.json to pick one",
                   "keep only the one to run, or fix checkpoints.json")


def read_run_config(run: Path) -> dict[str, Any]:
    path = run / "run_config" / "config.yaml"
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UseError(f"no {path}", "is this a run folder fetched by train/fetch.sh?") from None
    except (OSError, yaml.YAMLError) as e:
        raise UseError(f"cannot read {path}: {e}") from None
    if not isinstance(cfg, dict) or not isinstance(cfg.get("task"), dict):
        raise UseError(f"{path}: no task section", "is this a behavior_prompting training config?")
    return cfg


def uses_prompting(cfg: dict[str, Any]) -> bool:
    return bool((cfg["task"].get("shape_meta") or {}).get("use_prompting", False))


def training_dataset(run: Path, cfg: dict[str, Any]) -> Path:
    """The laptop copy of the dataset the run trained on: data/<folder>/package/<dataset>/<dataset>.zarr.zip."""
    trained = (cfg["task"].get("dataset") or {}).get("dataset_path")
    if not trained:
        raise UseError(f"{run / 'run_config' / 'config.yaml'}: no task.dataset.dataset_path",
                       "pass --prompt-dataset PATH")
    zip_name = Path(str(trained)).name                                 # pick1-v1.zarr.zip
    name = zip_name.removesuffix(".zarr.zip").removesuffix(".zarr")
    local = run.parent.parent / "package" / name / zip_name
    if not local.exists():
        raise UseError(f"the run trained on {zip_name}, not found at {local}",
                       "pull the package to the laptop, or pass --prompt-dataset PATH")
    return local


def _decode_vlen_utf8(chunk: bytes) -> list[str]:
    """numcodecs VLenUTF8: <u4 count, then per item <u4 length + utf-8 bytes."""
    (n,) = struct.unpack_from("<I", chunk, 0)
    out, pos = [], 4
    for _ in range(n):
        (size,) = struct.unpack_from("<I", chunk, pos)
        pos += 4
        out.append(chunk[pos:pos + size].decode("utf-8"))
        pos += size
    return out


def dataset_task_counts(dataset: Path) -> dict[str, int] | None:
    """{task name: number of demos} from meta/task_names of a .zarr.zip, or None if it cannot be read here (a
    compressed or multi-chunk array: the policy reads it properly at start-up)."""
    try:
        with zipfile.ZipFile(dataset) as z:
            meta = json.loads(z.read("meta/task_names/.zarray"))
            filters = meta.get("filters") or []
            if meta.get("compressor") is not None or [f.get("id") for f in filters] != ["vlen-utf8"]:
                return None
            names: list[str] = []
            n_chunks = -(-meta["shape"][0] // meta["chunks"][0])
            for i in range(n_chunks):
                names.extend(_decode_vlen_utf8(z.read(f"meta/task_names/{i}")))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, struct.error, UnicodeDecodeError):
        return None
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def choose(target: str, task: str | None, prompt_index: int | None, prompt_dataset: Path | None,
           prompt_split: str | None) -> Choice:
    if target == HOLD_STILL:
        if any(v is not None for v in (task, prompt_index, prompt_dataset, prompt_split)):
            raise UseError("hold_still takes no prompt options")
        return Choice(HOLD_STILL, {"kind": HOLD_STILL, "mock_kind": HOLD_STILL}, {}, [])

    run = Path(target).expanduser().absolute()
    if not run.is_dir():
        raise UseError(f"not a folder: {run}", "give a run folder, e.g. ~/Capstone/data/<folder>/policy/<run>, "
                                               "or hold_still")
    ckpt = find_checkpoint(run)
    cfg = read_run_config(run)
    # mock_kind too, so that `run --mock` runs this model against the fake Pi and fake PC
    policy: dict[str, Any] = {"kind": "diffusion", "mock_kind": "diffusion", "checkpoint": _home_path(ckpt)}
    notes: list[str] = []
    if not uses_prompting(cfg):
        if any(v is not None for v in (task, prompt_index, prompt_dataset, prompt_split)):
            raise UseError(f"{run.name} is a plain checkpoint (use_prompting is not true): it takes no prompt options")
        return Choice(run.name, policy, {}, notes)

    dataset = prompt_dataset.expanduser().absolute() if prompt_dataset else training_dataset(run, cfg)
    if not dataset.exists():
        raise UseError(f"prompt dataset not found: {dataset}")
    policy["prompt_dataset"] = _home_path(dataset)
    if prompt_split is not None:
        policy["prompt_split"] = prompt_split
    counts = dataset_task_counts(dataset)
    if task is None:
        if counts is None:
            raise UseError(f"cannot read the task names of {dataset.name} here", "pass --task NAME")
        if len(counts) != 1:
            raise UseError(f"{dataset.name} has {len(counts)} tasks: {', '.join(counts)}", "pass --task NAME")
        task = next(iter(counts))
    elif counts is not None and task not in counts:
        raise UseError(f"task {task!r} is not in {dataset.name}; it has {', '.join(counts)}")
    index = 0 if prompt_index is None else prompt_index
    if index < 0:
        raise UseError(f"--prompt-index {index}: must be 0 or more")
    if counts is not None:
        if index >= counts[task]:
            raise UseError(f"--prompt-index {index}: {dataset.name} has {counts[task]} demos of {task!r} (0-"
                           f"{counts[task] - 1})")
        notes.append(f"{dataset.name}: {counts[task]} demos of {task!r} in all; the policy checks prompt_index "
                     f"against its {prompt_split or 'train'} split at start-up")
    return Choice(run.name, policy, {"task_name": task, "prompt_index": index}, notes)


# ----------------------------------------------------------------------------------------- rewriting the station


def _section_span(lines: list[str], name: str) -> tuple[int, int] | None:
    """[start, end) of the top-level `name:` block: its key line and the indented or blank lines after it (trailing
    blank lines excluded). None if the file has no such key."""
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{name}:")), None)
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and (lines[end].strip() == "" or lines[end][0] in " \t"):
        end += 1
    while end > start + 1 and lines[end - 1].strip() == "":
        end -= 1
    return start, end


def _render(name: str, values: dict[str, Any], comment: str) -> list[str]:
    if not values:
        return [f"{name}: {{}}{' ' * max(1, 34 - len(name) - 4)}# {comment}\n"]
    body = yaml.safe_dump(values, default_flow_style=False, sort_keys=False, allow_unicode=True, width=1000)
    return [f"{name}:{' ' * max(1, 34 - len(name) - 1)}# {comment}\n"] + [f"  {line}\n" for line in body.splitlines()]


def rewrite(text: str, data: dict[str, Any], choice: Choice, speed: dict[str, dict[str, Any]]) -> str:
    """station.yaml text with the policy and manager sections replaced."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for name, choice_keys, chosen in (("policy", POLICY_CHOICE_KEYS, choice.policy),
                                      ("manager", MANAGER_CHOICE_KEYS, choice.manager)):
        old = data.get(name) or {}
        if not isinstance(old, dict):
            raise UseError(f"station.yaml: {name}: expected a map")
        values = {k: v for k, v in old.items() if k not in choice_keys}            # kept: speed and the rest
        values = {**chosen, **values, **speed[name]}
        if name == "policy":                                                       # kind and checkpoint first
            values = {**{k: values[k] for k in POLICY_CHOICE_KEYS if k in values}, **values}
        new = _render(name, values, f"set by `policy_deploy use` ({choice.label}, {stamp})")
        span = _section_span(lines, name)
        if span is None:
            lines.extend(new)
        else:
            lines[span[0]:span[1]] = new
    return "".join(lines)


def use(station_file: Path | None, target: str, *, task: str | None = None, prompt_index: int | None = None,
        prompt_dataset: Path | None = None, prompt_split: str | None = None, precision: str | None = None,
        diffusion_steps: int | None = None, exec_h: int | None = None) -> tuple[str, str, Choice]:
    """Rewrite the station file; returns (old text, new text, choice). Raises UseError or StationError."""
    path = Path(paths.default_station() if station_file is None else station_file).expanduser().absolute()
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
    except FileNotFoundError:
        raise UseError(f"station file not found: {path}") from None
    except (OSError, yaml.YAMLError) as e:
        raise UseError(f"cannot read {path}: {e}") from None
    choice = choose(target, task, prompt_index, prompt_dataset, prompt_split)
    if diffusion_steps is not None and diffusion_steps < 1:
        raise UseError(f"--diffusion-steps {diffusion_steps}: must be 1 or more", "8 is what this laptop needs")
    if exec_h is not None and not 1 <= exec_h <= ACTION_STEPS:
        raise UseError(f"--exec-h {exec_h}: must be 1 to {ACTION_STEPS}", "12 is what this laptop needs")
    speed = {"policy": {k: v for k, v in (("precision", precision), ("diffusion_steps", diffusion_steps))
                        if v is not None},
             "manager": {} if exec_h is None else {"exec_h": exec_h}}
    new_text = rewrite(text, data, choice, speed)

    tmp = path.with_name(f".{path.name}.use.tmp")
    try:
        tmp.write_text(new_text, encoding="utf-8")
        Station.load(tmp)                        # every key checked, as `check` and `run` will
        os.replace(tmp, path)
    except StationError as e:                    # name the real file, not the temporary copy
        raise StationError([p.replace(tmp.name, path.name) for p in e.problems]) from None
    finally:
        tmp.unlink(missing_ok=True)
    return text, new_text, choice


def diff(old: str, new: str, name: str) -> str:
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"{name} (before)", tofile=f"{name} (after)", n=1))


def frames_settled() -> bool | None:
    try:
        from deploy_contracts import TcpFrame
        return TcpFrame.load().settled
    except Exception:                            # deploy_contracts missing or frames.yaml broken: check says so
        return None


__all__ = ["UseError", "Choice", "use", "choose", "rewrite", "diff", "frames_settled", "StationError", "HOLD_STILL"]
