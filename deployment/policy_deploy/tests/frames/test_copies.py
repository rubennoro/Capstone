# adapted from rnoro/deployment:deployment/process_frames/tests/test_copies.py @ 91b178b
# changes: the copies live in policy_deploy/frames/_dataproc/ and tests/frames/ (the substitutions name the new
# import path and folder depth); the rig_pi_v2.yaml check is dropped (that rig file is not carried over: only
# rig_pi_v2_640.yaml, copied unchanged, is checked against rnoro/deployment instead).
"""Each COPIED file equals its source on rnoro/data-work, apart from the provenance header and the listed changes
(byte-identical bodies), and rigs/rig_pi_v2_640.yaml equals rnoro/deployment's file (STRUCTURE.md 2.1, 4.2).
Skipped when git or the source is not available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from frames_helpers import FOLDER

REPO = Path("/home/ruben/Capstone")
BRANCH = "rnoro/data-work"
SRC = "data-processing"
DEPLOY_REF = "91b178b"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def _available(ref: str) -> bool:
    if shutil.which("git") is None or not REPO.is_dir():
        return False
    try:
        _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


needs_branch = pytest.mark.skipif(not _available(BRANCH), reason=f"git or {BRANCH} not available")
needs_deploy = pytest.mark.skipif(not _available(DEPLOY_REF), reason=f"git or {DEPLOY_REF} not available")

PKG = "policy_deploy/frames/_dataproc/"
# (file here, source path, substitutions applied to the source)
COPIES = [
    (PKG + "imageprep.py", "dataproc/imageprep.py", []),
    (PKG + "config.py", "dataproc/config.py", [("from dataproc.paths import NAME_RE", "from .paths import NAME_RE")]),
    (PKG + "geometry/__init__.py", "dataproc/geometry/__init__.py", []),
    (PKG + "geometry/camera_model.py", "dataproc/geometry/camera_model.py", []),
    (PKG + "geometry/transforms.py", "dataproc/geometry/transforms.py", []),
    (PKG + "_vendor/cv_util.py", "dataproc/_vendor/cv_util.py", []),
    ("tests/frames/data/rig_v1.yaml", "configs/rigs/rig_v1.yaml", []),
    ("tests/frames/test_imageprep.py", "tests/test_imageprep.py", [
        ("from dataproc.", "from policy_deploy.frames._dataproc."),
        ('RIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "rigs" / "rig_v1.yaml"',
         'RIG_PATH = Path(__file__).resolve().parent / "data" / "rig_v1.yaml"'),
        ("REPO = Path(__file__).resolve().parents[2]", "REPO = Path(__file__).resolve().parents[4]"),
        ('vend = Path(__file__).resolve().parents[1] / "dataproc" / "_vendor" / "cv_util.py"',
         'vend = Path(__file__).resolve().parents[2] / "policy_deploy" / "frames" / "_dataproc" / "_vendor" / '
         '"cv_util.py"'),
        ('vend.read_text().split("\\n", 1)[1]', 'vend.read_text().split("\\n", 3)[3]')]),
    ("tests/frames/test_vendored_camera_model.py", "tests/test_vendored_camera_model.py",
     [("from dataproc.geometry.", "from policy_deploy.frames._dataproc.geometry.")]),
]


def _header_and_body(path: Path) -> tuple[list[str], str]:
    lines = path.read_text().split("\n")
    return lines[:2], "\n".join(lines[2:])


def _commit(src: str) -> str:
    return _git("log", "-1", "--format=%h", BRANCH, "--", f"{SRC}/{src}").strip()


@needs_branch
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


@needs_branch
def test_paths_keeps_only_name_re():
    header, body = _header_and_body(FOLDER / PKG / "paths.py")
    src = "dataproc/paths.py"
    assert header[0] == f"# from {BRANCH}:{SRC}/{src} @ {_commit(src)}"
    source_lines = _git("show", f"{BRANCH}:{SRC}/{src}").split("\n")
    kept = [ln for ln in body.split("\n") if ln.strip()]
    assert any(ln.startswith("NAME_RE = ") for ln in kept)
    for ln in kept:
        assert ln in source_lines, f"not a line of the source: {ln!r}"
    from policy_deploy.frames._dataproc.paths import NAME_RE
    assert NAME_RE.fullmatch("rig_pi_v2") and not NAME_RE.fullmatch("a b")


@needs_deploy
def test_rig_pi_v2_640_is_copied_unchanged():
    given = _git("show", f"{DEPLOY_REF}:deployment/process_frames/rigs/rig_pi_v2_640.yaml")
    assert (FOLDER / "rigs" / "rig_pi_v2_640.yaml").read_text() == given
