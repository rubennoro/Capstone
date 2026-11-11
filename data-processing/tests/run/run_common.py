"""Shared setup for the ``tests/run`` tests (no tests here).

Rendered once per pytest process (``--fast`` sim, the slow part): one calibration and :data:`N_DEMOS`
demonstrations, as contract episodes. :func:`pi_source` writes any of them in the Pi format under any directory
name and Pi episode number (different numbers = different recordings). :func:`calibrated_root` is a data root
whose folder ``block_raise`` was made by ``calibrate_task`` from ``episode_000000``; tests work on copies.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dataproc.calibrate import calibrate_task
from dataproc.paths import DataRoot, data_root
from dataproc.sim.episode import run_sim
from pi_fixtures import contract_to_pi, write_pi_manifest
from test_process_common import copy_root, quiet, tree_digest

__all__ = ["TASK", "RIG", "N_DEMOS", "pristine", "pi_source", "drop_truth", "calibrated_root", "copy_root",
           "quiet", "tree_digest", "standard_source"]

TASK = "block_raise"
RIG = "rig_v1_fast"
N_DEMOS = 4
_CACHE: dict = {}


def pristine(tmp_path_factory) -> dict:
    """``{"cal": dir, "demos": [dir, ...], "rigs": dir}``: the rendered contract episodes (read-only)."""
    if "pristine" not in _CACHE:
        base = tmp_path_factory.mktemp("run_sim")
        rigs = base / "rigs"
        run_sim(base / "sim", episodes=1, calibration=True, fast=True, rigs_dir=rigs, session_id="cal", log=quiet)
        run_sim(base / "sim", episodes=N_DEMOS, fast=True, rigs_dir=rigs, session_id="demo", log=quiet)
        eps = sorted((base / "sim").iterdir())
        _CACHE["pristine"] = {"cal": eps[0], "demos": eps[1:], "rigs": rigs}
    return _CACHE["pristine"]


def pi_source(contract_ep: Path, dst: Path, episode: int, **kw) -> Path:
    """``contract_ep`` in the Pi format at ``dst`` with Pi episode number ``episode``."""
    return contract_to_pi(contract_ep, dst, episode, with_exposure=True, **kw)


def drop_truth(pi_dir: Path) -> None:
    """Remove ``truth.npz`` (and rewrite the manifest): ``--slam oracle`` then fails at slam (``oracle_no_truth``)."""
    (pi_dir / "truth.npz").unlink()
    (pi_dir / "MANIFEST.sha256").unlink()
    write_pi_manifest(pi_dir)


def standard_source(tmp_path_factory, src: Path, bad: tuple[int, ...] = ()) -> Path:
    """``src/episode_000000`` (the calibration) and ``episode_00000k`` = demo k (k = 1..N_DEMOS); the demos in
    ``bad`` have no truth.npz."""
    p = pristine(tmp_path_factory)
    pi_source(p["cal"], src / "episode_000000", 0)
    for k, d in enumerate(p["demos"], start=1):
        pi_source(d, src / f"episode_{k:06d}", k)
        if k in bad:
            drop_truth(src / f"episode_{k:06d}")
    return src


def calibrated_root(tmp_path_factory) -> tuple[DataRoot, Path]:
    """``(root, source)``: :func:`standard_source` with demo 2 bad, and ``tasks/block_raise`` calibrated from
    ``episode_000000`` (shared, read-only: copy the root before changing it; the source is never changed)."""
    if "calibrated" not in _CACHE:
        base = tmp_path_factory.mktemp("run_calibrated")
        root = data_root(base / "root")
        root.rigs.mkdir(parents=True)
        shutil.copyfile(pristine(tmp_path_factory)["rigs"] / f"{RIG}.yaml", root.rigs / f"{RIG}.yaml")
        src = standard_source(tmp_path_factory, base / "src", bad=(2,))
        res = calibrate_task(root, TASK, "episode_000000", source=src, rig_id=RIG, log=quiet)
        assert res.ok and res.folder == TASK, res.message
        _CACHE["calibrated"] = (root, src)
    return _CACHE["calibrated"]
