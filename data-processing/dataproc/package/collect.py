"""Select what ``dataproc package`` puts into a dataset: the exported episodes of one or more task folders.

Per folder (in the order given), every ``episode_N`` (in N order) whose ``processed/status.json`` says
``exported`` and whose ``processed/episode.zarr`` exists is included. Everything else is listed as excluded, with
why: an ``episode_N`` that did not export, every ``failed/<pi_name>/`` (with its ``reason.json``) and every
``.staging/<pi_name>/`` ("not finished"). Nothing under ``failed/`` or ``.staging/`` is ever included.

Checks: every folder exists and has a ``task.yaml``; no folder is named twice; all folders have the same rig;
at least one episode overall; no Pi episode name twice (it is ``meta/episode_names`` in the zip); each
``episode.zarr``'s label is its folder's ``task.yaml`` label. Warning: a label with fewer than 2 episodes
(the prompting policy, dunetp, needs >= 2 per label). Read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dataproc.config import ConfigError, load_task_config
from dataproc.files import load_status, read_json
from dataproc.paths import DataRoot, episode_dirs, validate_name
from dataproc.report import episode_summary

__all__ = ["FAILED_DIR", "STAGING_DIR", "MIN_EPISODES_PER_LABEL", "PackageError", "FolderInfo", "PackagedEpisode",
           "ExcludedEntry", "Selection", "collect_episodes"]

FAILED_DIR = "failed"
STAGING_DIR = ".staging"
REASON_FILE = "reason.json"
#: The prompting policy (umi_policy_dunetp) samples its prompt from another demonstration of the same label.
MIN_EPISODES_PER_LABEL = 2


class PackageError(Exception):
    """``dataproc package`` cannot build the dataset (exit 1, nothing written)."""


@dataclass(frozen=True)
class FolderInfo:
    """One task folder named on the command line, with its ``task.yaml`` label and rig."""

    folder: str
    task_dir: Path
    label: str
    rig: str


@dataclass(frozen=True)
class PackagedEpisode:
    """One exported episode that goes into the dataset."""

    folder: str
    slot: str                   # episode_N
    pi_name: str                # the Pi episode dir name (== meta/episode_names in the zip)
    label: str                  # == meta/task_names in the zip
    n_steps: int
    calibration: str | None     # the calibration/<name> its width stage used
    zarr_path: Path             # tasks/<folder>/episode_N/processed/episode.zarr


@dataclass(frozen=True)
class ExcludedEntry:
    """Something in a task folder that is not packaged: ``entry`` is ``episode_N``, ``failed/<name>`` or
    ``.staging/<name>``; ``state`` is the episode status, ``failed`` or ``not finished``."""

    folder: str
    entry: str
    state: str
    reason: str


@dataclass
class Selection:
    """The result of :func:`collect_episodes`."""

    folders: list[FolderInfo]
    episodes: list[PackagedEpisode] = field(default_factory=list)
    excluded: list[ExcludedEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def labels(self) -> list[str]:
        """The labels of the included episodes, in first-seen order."""
        return list(dict.fromkeys(e.label for e in self.episodes))

    @property
    def rig(self) -> str:
        return self.folders[0].rig

    def counts_per_label(self) -> dict[str, int]:
        return {lab: sum(e.label == lab for e in self.episodes) for lab in self.labels}


# --------------------------------------------------------------------------------------------- the selection

def collect_episodes(root: DataRoot, folders: list[str]) -> Selection:
    """The episodes of ``folders`` (``tasks/<folder>``) to package, in folder order then ``episode_N`` order.
    Raises :class:`PackageError` when a check fails (see the module docstring)."""
    if not folders:
        raise PackageError("no task folder given")
    dup = sorted({f for f in folders if folders.count(f) > 1})
    if dup:
        raise PackageError(f"task folder named more than once: {', '.join(dup)}")
    sel = Selection([load_folder(root, f) for f in folders])
    check_same_rig(sel.folders)
    for info in sel.folders:
        collect_folder(info, sel)
    check_selection(sel)
    return sel


def load_folder(root: DataRoot, folder: str) -> FolderInfo:
    """``tasks/<folder>`` and its ``task.yaml`` (:class:`PackageError` if either is missing or invalid)."""
    try:
        validate_name(folder, "task folder")
    except ValueError as e:
        raise PackageError(str(e)) from None
    task_dir = root.tasks / folder
    if not task_dir.is_dir():
        raise PackageError(f"task folder {task_dir} does not exist")
    task_yaml = task_dir / "task.yaml"
    if not task_yaml.is_file():
        raise PackageError(f"{task_yaml} is missing")
    try:
        task = load_task_config(task_yaml)
    except (ConfigError, ValueError) as e:
        raise PackageError(str(e)) from None
    return FolderInfo(folder, task_dir, task.label, task.rig)


def check_same_rig(folders: list[FolderInfo]) -> None:
    rigs = {f.rig for f in folders}
    if len(rigs) > 1:
        detail = ", ".join(f"{f.folder}: {f.rig}" for f in folders)
        raise PackageError(f"the task folders were recorded with different rigs ({detail}); package them separately")


def collect_folder(info: FolderInfo, sel: Selection) -> None:
    """Adds ``info``'s exported episodes to ``sel.episodes`` and everything else to ``sel.excluded``."""
    for d in episode_dirs(info.task_dir):
        if not (d / "raw").is_dir():
            continue                       # an empty episode_N/ from an interrupted move (as in status)
        status = load_status(d / "processed")
        s = episode_summary(status)
        zarr_dir = d / "processed" / "episode.zarr"
        if s["status"] != "exported":
            sel.excluded.append(ExcludedEntry(info.folder, d.name, s["status"], "; ".join(s["reasons"]) or "-"))
        elif not zarr_dir.is_dir():
            sel.excluded.append(ExcludedEntry(info.folder, d.name, "exported", "processed/episode.zarr is missing"))
        else:
            sel.episodes.append(packaged_episode(info, d, status, s["n_steps"]))
    for name, reason in failed_entries(info.task_dir):
        sel.excluded.append(ExcludedEntry(info.folder, f"{FAILED_DIR}/{name}", "failed", reason))
    for name in staging_entries(info.task_dir):
        sel.excluded.append(ExcludedEntry(info.folder, f"{STAGING_DIR}/{name}", "not finished",
                                          "still in .staging (run dataproc run again)"))


def packaged_episode(info: FolderInfo, ep_dir: Path, status: dict, n_steps) -> PackagedEpisode:
    """The :class:`PackagedEpisode` of an exported ``episode_N``. Its name and label come from the
    ``episode.zarr`` attributes, which is what goes into the zip."""
    zarr_dir = ep_dir / "processed" / "episode.zarr"
    attrs = read_json(zarr_dir / ".zattrs", {})
    name, label = attrs.get("name"), attrs.get("task_label")
    if not name or not label:
        raise PackageError(f"{zarr_dir}: no name / task_label attributes (not a dataproc episode.zarr)")
    if label != info.label:
        raise PackageError(f"{zarr_dir}: label {label!r} != task.yaml label {info.label!r} of {info.folder} "
                           "(re-export it)")
    cal = ((status.get("width") or {}).get("outputs") or {}).get("calibration")
    return PackagedEpisode(info.folder, ep_dir.name, name, label, int(n_steps or 0), cal, zarr_dir)


def failed_entries(task_dir: Path) -> list[tuple[str, str]]:
    """``(pi_name, reason)`` of every ``failed/<pi_name>/``, by name. The reason is read from ``reason.json``
    (``stage``, ``reasons``, ``message``)."""
    out = []
    for d in _subdirs(task_dir / FAILED_DIR):
        rec = read_json(d / REASON_FILE, {})
        rec = rec if isinstance(rec, dict) else {}
        reasons = "; ".join(str(r) for r in (rec.get("reasons") or [])) or rec.get("message") or "no reason.json"
        out.append((d.name, f"{rec['stage']}: {reasons}" if rec.get("stage") else reasons))
    return out


def staging_entries(task_dir: Path) -> list[str]:
    """The names of the ``.staging/<pi_name>/`` dirs (recordings still being processed or waiting for SLAM)."""
    return [d.name for d in _subdirs(task_dir / STAGING_DIR)]


def _subdirs(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))


def check_selection(sel: Selection) -> None:
    """At least one episode; unique Pi names; a warning per label with fewer than 2 episodes."""
    if not sel.episodes:
        raise PackageError(f"no exported episode in {', '.join(f.folder for f in sel.folders)} "
                           "(run dataproc run first)")
    names = [e.pi_name for e in sel.episodes]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        where = "; ".join(f"{n}: " + ", ".join(f"{e.folder}/{e.slot}" for e in sel.episodes if e.pi_name == n)
                          for n in dup)
        raise PackageError(f"the same Pi episode is in the selection more than once ({where})")
    for label, n in sel.counts_per_label().items():
        if n < MIN_EPISODES_PER_LABEL:
            sel.warnings.append(f"label {label} has {n} episode(s); behavior prompting (umi_policy_dunetp) needs "
                                f">= {MIN_EPISODES_PER_LABEL} per label")
