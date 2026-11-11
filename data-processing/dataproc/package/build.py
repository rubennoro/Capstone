"""``package_dataset``: build one training ``.zarr.zip`` from the exported episodes of task folders.

Output, in the cluster's dataset layout (``~/CapstoneProject/datasets/<task>/<dataset>/<dataset>.zarr.zip``,
scripts/run_pipeline.sh)::

    $DATAPROC_ROOT/datasets/<label>/<DATASET>/<DATASET>.zarr.zip     one label
    $DATAPROC_ROOT/datasets/multi/<DATASET>/<DATASET>.zarr.zip       several labels
                                             package.json            what went in (see :func:`package_record`)

The zip is written by :func:`dataproc.export.zarr_writer.build_task_zarr_zip` (unchanged: concatenates the
per-episode zarrs, re-offsets the ultrawide index maps, temp file + rename in the dataset directory, so a failed
build never leaves a partial zip). An existing dataset directory is refused unless ``overwrite``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dataproc import __version__
from dataproc.files import iso_utc, write_json
from dataproc.fsutil import rmdir_if_empty, sha256_file
from dataproc.package.collect import PackageError, Selection, collect_episodes
from dataproc.paths import DataRoot, validate_name

__all__ = ["DATASETS_DIR", "MULTI_LABEL", "PACKAGE_FILE", "PackageResult", "dataset_label", "dataset_dir",
           "package_dataset", "package_record"]

DATASETS_DIR = "datasets"
MULTI_LABEL = "multi"
PACKAGE_FILE = "package.json"


@dataclass
class PackageResult:
    """What :func:`package_dataset` wrote: the zip, its ``package.json`` record and the selection behind it."""

    dataset: str
    label: str                  # the first directory level: the task label, or ``multi``
    zip_path: Path
    record: dict                # == package.json
    selection: Selection

    @property
    def size_bytes(self) -> int:
        return self.zip_path.stat().st_size


def dataset_label(labels: list[str]) -> str:
    """The first directory level under ``datasets/``: the label, or ``multi`` when there are several."""
    return labels[0] if len(labels) == 1 else MULTI_LABEL


def dataset_dir(root: DataRoot, dataset: str, label: str) -> Path:
    """``$ROOT/datasets/<label>/<dataset>``."""
    return root.path / DATASETS_DIR / label / dataset


def package_dataset(root: DataRoot, dataset: str, folders: list[str], overwrite: bool = False,
                    log: Callable = print) -> PackageResult:
    """Collect the exported episodes of ``folders`` (:func:`~dataproc.package.collect.collect_episodes`), write
    ``datasets/<label>/<dataset>/<dataset>.zarr.zip`` and its ``package.json``. Raises :class:`PackageError`
    (nothing written) on a bad name, a failed check, an existing dataset without ``overwrite``, or a failed build
    (an overwritten dataset then keeps its previous zip and package.json)."""
    try:
        validate_name(dataset, "dataset name")
    except ValueError as e:
        raise PackageError(str(e)) from None
    sel = collect_episodes(root, list(folders))
    label = dataset_label(sel.labels)
    out_dir = dataset_dir(root, dataset, label)
    zip_path = out_dir / f"{dataset}.zarr.zip"
    if out_dir.exists() and not overwrite:
        raise PackageError(f"dataset {out_dir} already exists (pick another name, or pass --overwrite)")
    log(f"building {zip_path} from {len(sel.episodes)} episodes ...")
    build_zip(sel, out_dir, zip_path)
    record = package_record(dataset, label, sel, zip_path)
    write_json(out_dir / PACKAGE_FILE, record)
    return PackageResult(dataset, label, zip_path, record, sel)


def build_zip(sel: Selection, out_dir: Path, zip_path: Path) -> dict:
    """``build_task_zarr_zip`` of the selected ``episode.zarr``s, in order, building inside ``out_dir`` (one
    filesystem for the final rename). A directory this call created is removed again if the build fails."""
    from dataproc.export.zarr_writer import build_task_zarr_zip

    created = [p for p in (out_dir.parent.parent, out_dir.parent, out_dir) if not p.exists()]
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return build_task_zarr_zip([e.zarr_path for e in sel.episodes], zip_path, tmp_dir=out_dir)
    except Exception as e:  # noqa: BLE001 - reported as a package error
        for p in reversed(created):
            rmdir_if_empty(p)
        raise PackageError(f"building {zip_path.name} failed ({type(e).__name__}: {e})") from e


def package_record(dataset: str, label: str, sel: Selection, zip_path: Path) -> dict:
    """The ``package.json`` content: dataset, label dir, folders, per episode (folder, slot, pi_name, label,
    n_steps, calibration), totals, rig, the excluded entries, dataproc version, zip name + size + sha256, time."""
    return {
        "dataset": dataset,
        "label_dir": label,
        "labels": sel.labels,
        "folders": [{"folder": f.folder, "label": f.label, "rig": f.rig} for f in sel.folders],
        "rig": sel.rig,
        "episodes": [{"folder": e.folder, "slot": e.slot, "pi_name": e.pi_name, "label": e.label,
                      "n_steps": e.n_steps, "calibration": e.calibration} for e in sel.episodes],
        "excluded": [{"folder": x.folder, "entry": x.entry, "state": x.state, "reason": x.reason}
                     for x in sel.excluded],
        "totals": {"episodes": len(sel.episodes), "n_steps": sum(e.n_steps for e in sel.episodes),
                   "per_label": sel.counts_per_label(), "excluded": len(sel.excluded)},
        "warnings": list(sel.warnings),
        "zip": {"name": zip_path.name, "size_bytes": zip_path.stat().st_size, "sha256": sha256_file(zip_path)},
        "dataproc_version": __version__,
        "created": iso_utc(time.time()),
    }
