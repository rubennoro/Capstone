"""``umiproc package DATASET FOLDER [FOLDER ...] [--episodes N] [--root DIR] [--overwrite]``: combine the finished episodes of task
folders into one training ``.zarr.zip`` (the replay buffer ``UmiTaskDataset`` reads, for ``umi_policy_dunet`` and
``umi_policy_dunetp``).

The work is the packager copied from dataproc (:mod:`umiproc.dp.package`, unchanged): every ``episode_N`` whose
``processed/status.json`` says ``exported`` goes in, in N order; ``failed/`` and ``.staging/`` never do. ``--episodes N`` keeps only the first N of those (folder order, then N
order), so e.g. 10/15/20/25 are nested subsets for a data-scaling comparison. Output::

    $UMIPROC_ROOT/datasets/<label>/DATASET/DATASET.zarr.zip     (datasets/multi/... for several labels)
                                          package.json           what went in, with the zip's sha256

Exit codes: 0 written; 1 nothing written (the message says why: no exported episode, a name already taken, ...).
"""

from __future__ import annotations

import argparse
import sys

__all__ = ["add_arguments", "run"]


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("dataset", metavar="DATASET", help="a new dataset name, e.g. pick1-v1 (letters, digits, '_', '-')")
    p.add_argument("folders", metavar="FOLDER", nargs="+", help="task folders under tasks/ whose episodes go in")
    p.add_argument("--episodes", type=int, default=None, metavar="N",
                   help="only the first N exported episodes (recording order); default all")
    p.add_argument("--root", default=None, metavar="DIR", help="data root (default $UMIPROC_ROOT or ~/data/umi)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing dataset of that name")


def run(args) -> int:
    from umiproc.dp.package.cli import cli_package
    from umiproc.dp.paths import data_root

    try:
        root = data_root(args.root)
    except ValueError as e:
        print(f"umiproc package: {e}", file=sys.stderr)
        return 1
    print(f"package: data root {root.path}", flush=True)
    return cli_package(root, args.dataset, args.folders, overwrite=args.overwrite,
                       max_episodes=args.episodes)
