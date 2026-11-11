"""``dataproc package DATASET FOLDER [FOLDER ...] [--overwrite]``: printing and exit codes."""

from __future__ import annotations

import sys
import traceback

from dataproc.package.build import PackageResult, package_dataset
from dataproc.package.collect import PackageError, Selection
from dataproc.paths import DataRoot
from dataproc.report import format_table

__all__ = ["cli_package", "format_selection", "cluster_handoff"]

XFER_HOST = "<you>@xfer.discovery.neu.edu"


def format_selection(sel: Selection) -> str:
    """The included and the excluded entries, one table."""
    rows = [[e.folder, e.slot, e.pi_name, e.label, e.n_steps, "included"] for e in sel.episodes]
    rows += [[x.folder, x.entry, "-", "-", "-", f"excluded ({x.state}): {x.reason}"] for x in sel.excluded]
    return format_table(rows, ["folder", "entry", "pi_name", "label", "n_steps", "packaged"])


def cluster_handoff(res: PackageResult) -> list[str]:
    """The commands that put the zip where the cluster expects it and start training on it."""
    remote_dir = f"CapstoneProject/datasets/{res.label}/{res.dataset}"
    return ["To train on the cluster:",
            f"  ssh {XFER_HOST} mkdir -p {remote_dir}",
            f"  rsync -av {res.zip_path} {XFER_HOST}:{remote_dir}/{res.dataset}.zarr.zip",
            f"  scripts/run_pipeline.sh train --task {res.label} --dataset {res.dataset} --note \"...\"   "
            "(on the laptop)"]


def _summary(res: PackageResult) -> str:
    t = res.record["totals"]
    per_label = ", ".join(f"{k} {v}" for k, v in t["per_label"].items())
    lines = [format_selection(res.selection), ""]
    lines += [f"warning: {w}" for w in res.selection.warnings]
    lines += [f"wrote {res.zip_path} ({res.size_bytes / 1e6:.1f} MB; {t['episodes']} episodes, {t['n_steps']} steps;"
              f" {per_label}; rig {res.record['rig']})",
              f"      {res.zip_path.parent / 'package.json'}", ""]
    return "\n".join(lines + cluster_handoff(res))


def cli_package(root: DataRoot, dataset: str, folders: list[str], overwrite: bool = False) -> int:
    """``dataproc package``; returns the exit code (0 written, 1 on any error, message on stderr)."""
    try:
        res = package_dataset(root, dataset, folders, overwrite=overwrite, log=print)
    except PackageError as e:
        print(f"dataproc package: {e}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - unexpected: traceback, exit 1
        traceback.print_exc()
        return 1
    print(_summary(res))
    return 0
