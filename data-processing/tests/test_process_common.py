"""Shared helpers for the ``test_process_*`` tests (no tests here).

``template_root(tmp_path_factory)`` builds, once per pytest process, a data root with the ``--fast`` sim rig:
1 calibration + 3 demonstrations, intaken into ``tasks/block_raise`` (the slow part, ~45 s). ``processed_root``
also runs ``process --slam oracle`` once on a copy. Tests work on copies (:func:`copy_root`), never on these.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import yaml

from dataproc.intake import run_intake
from dataproc.paths import DataRoot, data_root
from dataproc.process import run_process
from dataproc.sim.episode import fast_rig, generate_episode, run_sim

TASK = "block_raise"
N_DEMOS = 3
RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"
_CACHE: dict = {}


def quiet(*a, **k):
    pass


def _build_template(base: Path) -> DataRoot:
    root = data_root(base)
    run_sim(root.incoming, episodes=1, calibration=True, fast=True, rigs_dir=root.rigs, session_id="cal", log=quiet)
    assert run_intake(root, TASK, calibration=True, log=quiet).by_status("ingested")
    run_sim(root.incoming, episodes=N_DEMOS, fast=True, rigs_dir=root.rigs, session_id="demo", log=quiet)
    assert len(run_intake(root, TASK, log=quiet).by_status("ingested")) == N_DEMOS
    return root


def template_root(tmp_path_factory) -> DataRoot:
    """Sim + intake, not processed (shared, read-only)."""
    if "template" not in _CACHE:
        _CACHE["template"] = _build_template(tmp_path_factory.mktemp("process_template"))
    return _CACHE["template"]


def processed_root(tmp_path_factory) -> tuple[DataRoot, object]:
    """``template_root`` processed once with the oracle (shared, read-only): ``(root, ProcessResult)``."""
    if "processed" not in _CACHE:
        root = copy_root(template_root(tmp_path_factory), tmp_path_factory.mktemp("process_done") / "root")
        _CACHE["processed"] = (root, run_process(root, TASK, slam="oracle", log=quiet))
    return _CACHE["processed"]


def copy_root(src: DataRoot, dst: Path) -> DataRoot:
    shutil.copytree(src.path, dst, symlinks=True)
    return data_root(dst)


def task_dir(root: DataRoot, folder: str = TASK) -> Path:
    return root.tasks / folder


def fast_rig_dict() -> dict:
    return fast_rig(RIG_PATH)[0]


def write_rig(root: DataRoot, d: dict) -> None:
    (root.rigs / f"{d['rig_id']}.yaml").write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")


def tree_digest(path: Path) -> dict:
    """``{relative path: sha256}`` of every file under ``path`` (also dot-files)."""
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def add_demo(root: DataRoot, name: str, seed: int, *, duration_s: float = 5.0, faults=None) -> None:
    """Render one ``--fast`` demonstration into ``incoming/<name>`` and intake it into the task."""
    _, rig = fast_rig(RIG_PATH)
    generate_episode(root.incoming / name, rig, "demonstration", seed=seed, duration_s=duration_s, preroll_s=2.0,
                     session_id=name, episode_number=0, faults=faults)
    res = run_intake(root, TASK, log=quiet)
    assert [o.status for o in res.outcomes] == ["ingested"], [o.describe() for o in res.outcomes]


def T_from(pos, rotvec) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_rotvec(np.asarray(rotvec, np.float64)).as_matrix()
    T[:, :3, 3] = pos
    return T
