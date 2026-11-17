"""Helpers for the ``test_umiproc_*`` tests (not a test module): tiny Pi-format recordings built with the copied test
fixtures (``tests/dp/fixtures.py`` + ``umiproc.dp.pi_format.contract_to_pi``), and a digest of a folder tree."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path

from umiproc.dp.pi_format import contract_to_pi

_FIXTURES = Path(__file__).resolve().parent / "dp" / "fixtures.py"


def _dp_fixtures():
    spec = importlib.util.spec_from_file_location("umiproc_dp_test_fixtures", _FIXTURES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fx = _dp_fixtures()


def pi_recording(source: Path, number: int, *, n_cam0: int = 12, n_cam1: int = 6, complete: bool = True,
                 seed: int | None = None) -> Path:
    """A valid Pi-format recording ``source/episode_<number:06d>`` (tiny videos); ``seed`` changes its IMU noise
    (another recording with the same Pi episode number)."""
    work = Path(source).parent / f".contract-{Path(source).name}-{number}-{seed}"
    contract = _fx.build_episode(work, n_cam0=n_cam0, n_cam1=n_cam1, episode_number=number if seed is None else seed)
    return contract_to_pi(contract, Path(source) / f"episode_{number:06d}", number, complete=complete)


def tree_digest(path: Path) -> dict:
    """``{relative path: sha256 or 'dir' or 'link -> target'}`` of everything under ``path``."""
    path = Path(path)
    out = {}
    if not path.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(path):
        for n in dirnames + filenames:
            p = Path(dirpath) / n
            rel = str(p.relative_to(path))
            if p.is_symlink():
                out[rel] = f"link -> {os.readlink(p)}"
            elif p.is_dir():
                out[rel] = "dir"
            else:
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
