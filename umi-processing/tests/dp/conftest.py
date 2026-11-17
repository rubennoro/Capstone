"""Puts ``umi-processing/`` on ``sys.path`` so the tests run without installing the package, and keeps every test away
from the real data roots: ``$UMIPROC_ROOT`` is a temporary folder for the whole session unless a test sets it itself
(set before any module fixture runs), and ``$DATAPROC_ROOT`` is removed (umiproc never reads it, except to refuse it
as its root)."""

import sys
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


@pytest.fixture(autouse=True, scope="session")
def _never_the_real_root(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("UMIPROC_ROOT", str(tmp_path_factory.mktemp("umiproc_root")))
        mp.delenv("DATAPROC_ROOT", raising=False)
        yield
