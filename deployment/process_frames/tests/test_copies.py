"""Each COPIED file equals its source on rnoro/data-work, apart from the provenance header and the listed changes
(STRUCTURE.md §3). Skipped when git or the source branch is not available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from process_frames import FOLDER
from process_frames._dataproc.config import parse_rig

REPO = Path("/home/ruben/Capstone")
BRANCH = "rnoro/data-work"
SRC = "data-processing"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def _available() -> bool:
    if shutil.which("git") is None or not REPO.is_dir():
        return False
    try:
        _git("rev-parse", "--verify", BRANCH)
    except subprocess.CalledProcessError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason=f"git or {BRANCH} not available")

PKG = "process_frames/_dataproc/"
# (file here, source path, substitutions applied to the source)
COPIES = [
    (PKG + "imageprep.py", "dataproc/imageprep.py", []),
    (PKG + "config.py", "dataproc/config.py", [("from dataproc.paths import NAME_RE", "from .paths import NAME_RE")]),
    (PKG + "geometry/__init__.py", "dataproc/geometry/__init__.py", []),
    (PKG + "geometry/camera_model.py", "dataproc/geometry/camera_model.py", []),
    (PKG + "geometry/transforms.py", "dataproc/geometry/transforms.py", []),
    (PKG + "_vendor/cv_util.py", "dataproc/_vendor/cv_util.py", []),
    ("tests/data/rig_v1.yaml", "configs/rigs/rig_v1.yaml", []),
    ("tests/test_imageprep.py", "tests/test_imageprep.py", [
        ("from dataproc.", "from process_frames._dataproc."),
        ('RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"',
         'RIG_PATH = Path(__file__).resolve().parent / "data" / "rig_v1.yaml"'),
        ("REPO = Path(__file__).resolve().parents[2]", "REPO = Path(__file__).resolve().parents[3]"),
        ('vend = Path(__file__).resolve().parents[1] / "dataproc" / "_vendor" / "cv_util.py"',
         'vend = Path(__file__).resolve().parents[1] / "process_frames" / "_dataproc" / "_vendor" / "cv_util.py"'),
        ('vend.read_text().split("\\n", 1)[1]', 'vend.read_text().split("\\n", 3)[3]')]),
    ("tests/test_vendored_camera_model.py", "tests/test_vendored_camera_model.py",
     [("from dataproc.geometry.", "from process_frames._dataproc.geometry.")]),
]


def _header_and_body(path: Path) -> tuple[list[str], str]:
    lines = path.read_text().split("\n")
    return lines[:2], "\n".join(lines[2:])


def _commit(src: str) -> str:
    return _git("log", "-1", "--format=%h", BRANCH, "--", f"{SRC}/{src}").strip()


@pytest.mark.parametrize("here,src,subs", COPIES, ids=[c[0] for c in COPIES])
def test_copy_equals_source(here, src, subs):
    header, body = _header_and_body(FOLDER / here)
    assert header[0] == f"# from {BRANCH}:{SRC}/{src} @ {_commit(src)}"
    assert header[1].startswith("# changes: ")
    if not subs:
        assert header[1] == "# changes: none"
    expected = _git("show", f"{BRANCH}:{SRC}/{src}")
    for old, new in subs:
        assert old in expected, f"listed change no longer applies: {old!r}"
        expected = expected.replace(old, new)
    assert body == expected


def test_paths_keeps_only_name_re():
    header, body = _header_and_body(FOLDER / PKG / "paths.py")
    src = "dataproc/paths.py"
    assert header[0] == f"# from {BRANCH}:{SRC}/{src} @ {_commit(src)}"
    source_lines = _git("show", f"{BRANCH}:{SRC}/{src}").split("\n")
    kept = [ln for ln in body.split("\n") if ln.strip()]
    assert any(ln.startswith("NAME_RE = ") for ln in kept)
    for ln in kept:
        assert ln in source_lines, f"not a line of the source: {ln!r}"
    from process_frames._dataproc.paths import NAME_RE
    assert NAME_RE.fullmatch("rig_pi_v1_standin") and not NAME_RE.fullmatch("a b")


def _plain(rig) -> dict:
    """The parsed rig as comparable plain values (arrays as lists)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return np.round(v, 12).tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if hasattr(v, "__dataclass_fields__"):
            return {k: conv(getattr(v, k)) for k in v.__dataclass_fields__}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        return v
    return conv(rig)


def test_standin_rig_equals_rig_pi_v1_except_the_stated_changes():
    """Four changes: header comments, rig_id, main fps 60 -> 30, the tcp block (T_cam_tcp from deploy.geometry)."""
    src = "configs/rigs/rig_pi_v1.yaml"
    source = parse_rig(yaml.safe_load(_git("show", f"{BRANCH}:{SRC}/{src}")), "rig_pi_v1.yaml")
    standin_path = FOLDER / "rigs" / "rig_pi_v1_standin.yaml"
    standin = parse_rig(yaml.safe_load(standin_path.read_text()), "rig_pi_v1_standin.yaml")
    a, b = _plain(source), _plain(standin)
    assert a["rig_id"] == "rig_pi_v1" and b["rig_id"] == "rig_pi_v1_standin"
    assert a["cameras"]["main"]["fps"] == 60.0 and b["cameras"]["main"]["fps"] == 30.0
    for d in (a, b):
        for key in ("rig_id", "source_path", "T_cam_tcp"):
            d.pop(key)
        d["cameras"]["main"].pop("fps")
    assert a == b
    assert standin_path.read_text().splitlines()[1].startswith(f"# Copied from {BRANCH}:{SRC}/{src} @ ")
