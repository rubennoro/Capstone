# adapted from rnoro/deployment:deployment/process_frames/tests/test_copies.py @ 91b178b
# changes: the copies live in policy_deploy/frames/_dataproc/ and tests/frames/ (the substitutions name the new
# import path and folder depth); the rig_pi_v2.yaml check is dropped. 2026-11-19: config.py is copied from training's
# processor (umi-processing on rnoro/data-work-auto) instead; the deployment rig is generated
# (../contracts/rigs/rig_pi_v3_640.yaml, frames/deploy_rig.py), so its check is "matches its sources".
"""Each COPIED file equals its source (rnoro/data-work, or umi-processing on rnoro/data-work-auto for config.py), apart
from the provenance header and the listed changes (byte-identical bodies), and the contract rig file is what
deploy_rig makes from training's rig + lens now (STRUCTURE.md 2.1, 4.2). Skipped when git or a source is missing."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from frames_helpers import FOLDER

REPO = Path("/home/ruben/Capstone")
BRANCH = "rnoro/data-work"
SRC = "data-processing"
UMIPROC_BRANCH = "rnoro/data-work-auto"           # training's processor: the parser of the rig files training read
UMIPROC_SRC = "umi-processing"
TRAINING_RIG = Path("~/data/umi/rigs/rig_pi_v3.yaml").expanduser()


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
needs_umiproc = pytest.mark.skipif(not _available(UMIPROC_BRANCH), reason=f"git or {UMIPROC_BRANCH} not available")
needs_training_rig = pytest.mark.skipif(not TRAINING_RIG.is_file(), reason=f"{TRAINING_RIG} not on this machine")

PKG = "policy_deploy/frames/_dataproc/"
# (file here, source path, substitutions applied to the source)
COPIES = [
    (PKG + "imageprep.py", "dataproc/imageprep.py", []),
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
UMIPROC_COPIES = [
    (PKG + "config.py", "umiproc/dp/config.py",
     [("from umiproc.dp.paths import NAME_RE", "from .paths import NAME_RE")]),
]


def _header_and_body(path: Path) -> tuple[list[str], str]:
    lines = path.read_text().split("\n")
    return lines[:2], "\n".join(lines[2:])


def _commit(src: str, branch: str = BRANCH, root: str = SRC) -> str:
    return _git("log", "-1", "--format=%h", branch, "--", f"{root}/{src}").strip()


def _check_copy(here: str, src: str, subs, branch: str, root: str) -> None:
    header, body = _header_and_body(FOLDER / here)
    assert header[0] == f"# from {branch}:{root}/{src} @ {_commit(src, branch, root)}"
    assert header[1].startswith("# changes: ")
    if not subs:
        assert header[1] == "# changes: none"
    expected = _git("show", f"{branch}:{root}/{src}")
    for old, new in subs:
        assert old in expected, f"listed change no longer applies: {old!r}"
        expected = expected.replace(old, new)
    assert body == expected


@needs_branch
@pytest.mark.parametrize("here,src,subs", COPIES, ids=[c[0] for c in COPIES])
def test_copy_equals_source(here, src, subs):
    _check_copy(here, src, subs, BRANCH, SRC)


@needs_umiproc
@pytest.mark.parametrize("here,src,subs", UMIPROC_COPIES, ids=[c[0] for c in UMIPROC_COPIES])
def test_copy_equals_umiproc_source(here, src, subs):
    _check_copy(here, src, subs, UMIPROC_BRANCH, UMIPROC_SRC)


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


@needs_training_rig
def test_contract_rig_matches_its_sources(capsys):
    """../contracts/rigs/rig_pi_v3_640.yaml is what deploy_rig makes from training's rig + lens today."""
    from policy_deploy.frames.deploy_rig import main
    assert main(["--check"]) == 0, capsys.readouterr().out
