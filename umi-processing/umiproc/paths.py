"""Where umiproc's files live: the repo's ``configs/`` and the data root (docs/DESIGN.md "Where it runs and the data
root").

* :data:`REPO` is the ``umi-processing/`` folder; :func:`repo_config` names a file under its ``configs/``.
* :func:`data_root` is the output root: ``~/data/umi`` unless ``--root`` or ``$UMIPROC_ROOT`` says otherwise. It is
  the **only** source of the root: the copied dataproc code (``umiproc.dp.paths.data_root``) asks it too. Refused:

  - ``/data`` itself: its ``incoming/`` would be the Pi's ``/data/incoming``, which umiproc only reads;
  - the Pi's ``/data/incoming`` and any folder inside it: umiproc would write ``.run.lock``, ``incoming/``,
    ``tasks/`` and ``pi_sources.csv`` into the Pi's delivery folder;
  - dataproc's own root (``~/data``, and ``$DATAPROC_ROOT`` when set): umiproc's ``tasks/``, ``incoming/``,
    ``rigs/`` and ``pi_sources.csv`` would mix with dataproc's. umiproc never reads ``$DATAPROC_ROOT`` otherwise;
  - any folder inside dataproc's managed subfolders (``incoming``, ``tasks``, ``rigs``, ``quarantine``,
    ``datasets`` of ``~/data``, and of ``$DATAPROC_ROOT`` when set). Another folder inside dataproc's root, such as
    the default ``~/data/umi``, is fine.

  Every comparison is made on the normalised path and on its realpath (symlinks resolved), on both sides.

* The data-root copies the processing reads (``rigs/``): :func:`rig_file`, :func:`rig_slam_file`,
  :func:`rig_lens_file`.

Pure path logic: nothing here creates, moves or deletes files.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO", "ENV_ROOT", "DEFAULT_ROOT", "FORBIDDEN_ROOTS", "DATAPROC_ENV_ROOT", "DATAPROC_DEFAULT_ROOT",
           "PI_INCOMING", "DATAPROC_MANAGED", "repo_config", "data_root", "check_root", "forbidden_roots",
           "forbidden_trees", "lens_dir", "rig_file", "rig_slam_file", "rig_lens_file"]

#: The ``umi-processing/`` folder (this file is ``umi-processing/umiproc/paths.py``).
REPO = Path(__file__).resolve().parents[1]
ENV_ROOT = "UMIPROC_ROOT"
DEFAULT_ROOT = Path.home() / "data" / "umi"
#: Roots that are always refused (compared after resolving symlinks and ``..``).
FORBIDDEN_ROOTS = (Path("/data"),)
#: dataproc's root (data-processing/): ``$DATAPROC_ROOT``, default ``~/data``. Only ever refused as umiproc's root.
DATAPROC_ENV_ROOT = "DATAPROC_ROOT"
DATAPROC_DEFAULT_ROOT = Path.home() / "data"
#: The Pi's delivery folder (only ever read): refused as the root, and so is every folder inside it.
PI_INCOMING = Path("/data/incoming")
#: dataproc's managed subfolders of its root: refused as the root, and so is every folder inside them.
DATAPROC_MANAGED = ("incoming", "tasks", "rigs", "quarantine", "datasets")


def repo_config(*parts: str) -> Path:
    """``umi-processing/configs/<parts...>``, e.g. ``repo_config("rigs", "rig_pi_v3.yaml")``."""
    return REPO.joinpath("configs", *parts)


def _abs(path: str | os.PathLike) -> Path:
    return Path(os.path.normpath(Path(path).expanduser().absolute()))


def _dataproc_roots() -> list[tuple[Path, str]]:
    """dataproc's roots: ``~/data``, and ``$DATAPROC_ROOT`` when set (read here, and only here)."""
    out = [(DATAPROC_DEFAULT_ROOT, "dataproc's data root (data-processing/, default ~/data)")]
    env = os.environ.get(DATAPROC_ENV_ROOT)
    if env:
        out.append((_abs(env), f"dataproc's data root (${DATAPROC_ENV_ROOT})"))
    return out


def forbidden_roots() -> list[tuple[Path, str]]:
    """``(path, why)`` of every root umiproc refuses as itself (a folder inside it may be fine)."""
    out = [(bad, f"its incoming/ would be the Pi's {bad}/incoming (only ever read)") for bad in FORBIDDEN_ROOTS]
    out += [(root, f"it is {what}") for root, what in _dataproc_roots()]
    return out


def forbidden_trees() -> list[tuple[Path, str]]:
    """``(path, why)`` of every folder umiproc refuses as its root **and** refuses every folder inside of."""
    out = [(PI_INCOMING, "it is the Pi's delivery folder (only ever read)")]
    for root, what in _dataproc_roots():
        out += [(root / sub, f"it is a folder dataproc manages ({sub}/ of {what})") for sub in DATAPROC_MANAGED]
    return out


def _forms(p: Path) -> set[Path]:
    return {p, Path(os.path.realpath(p))}


def check_root(path: str | os.PathLike) -> Path:
    """``path`` (absolute, normalised) if umiproc may use it as its data root, else ``ValueError``."""
    p = _abs(path)
    mine = _forms(p)
    hint = f"Use the default {DEFAULT_ROOT}, or another folder with --root or ${ENV_ROOT}."
    for bad, why in forbidden_roots():
        if mine & _forms(bad):
            raise ValueError(f"data root {p} is refused: it resolves to {bad}, and {why}. {hint}")
    for bad, why in forbidden_trees():
        for b in _forms(bad):
            if any(m == b or m.is_relative_to(b) for m in mine):
                raise ValueError(f"data root {p} is refused: it is {bad} or inside it, and {why}. {hint}")
    return p


def data_root(path: str | os.PathLike | None = None) -> Path:
    """The data root: ``path`` if given, else ``$UMIPROC_ROOT`` if set and non-empty, else ``~/data/umi``.

    Returned absolute (``~`` expanded). Raises ``ValueError`` when it resolves to a refused root (``/data``,
    ``/data/incoming`` or inside it, dataproc's root or inside its managed subfolders; see the module docstring).
    """
    if path is None or str(path) == "":
        path = os.environ.get(ENV_ROOT) or DEFAULT_ROOT
    return check_root(path)


def lens_dir(root: Path, pi_name: str) -> Path:
    """``<root>/lens/<pi_name>/``: the results of ``umiproc lens-calibrate`` for one recording."""
    from umiproc.dp.paths import validate_episode_dir_name

    return Path(root) / "lens" / validate_episode_dir_name(pi_name)


def _rigs_file(root: Path, rig_id: str, suffix: str) -> Path:
    from umiproc.dp.paths import validate_name

    return Path(root) / "rigs" / f"{validate_name(rig_id, 'rig id')}{suffix}"


def rig_file(root: Path, rig_id: str) -> Path:
    """``<root>/rigs/<rig_id>.yaml``: the data-root copy of the rig file (``configs/rigs/<rig_id>.yaml``)."""
    return _rigs_file(root, rig_id, ".yaml")


def rig_slam_file(root: Path, rig_id: str) -> Path:
    """``<root>/rigs/<rig_id>.slam.yaml``: the data-root copy of the SLAM config (``configs/slam/<rig_id>.yaml``)."""
    return _rigs_file(root, rig_id, ".slam.yaml")


def rig_lens_file(root: Path, rig_id: str) -> Path:
    """``<root>/rigs/<rig_id>.lens.yaml``: the applied lens calibration the processing reads (optional)."""
    return _rigs_file(root, rig_id, ".lens.yaml")
