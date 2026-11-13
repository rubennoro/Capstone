"""Each COPIED file equals its source on rnoro/data-work, apart from the provenance header and the listed changes,
and the real rig file rig_pi_v2.yaml equals the user's file as given apart from its one stated change (STRUCTURE.md
§3). Skipped when git or the source is not available."""

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


#: The commit that added rig_pi_v2.yaml exactly as the user gave it, and the one change made to it since.
V2_SOURCE_COMMIT = "196eae4"
V2_PATH = "deployment/process_frames/rigs/rig_pi_v2.yaml"
_PAD = " " * 54
V2_CHANGE = (
    "  max_opening_m: 0.110" + " " * 32 + "# max_width - min_width of the fit (tag-to-tag x, metres)\n",
    "  max_opening_m: 0.080" + " " * 32 + "# max_width - min_width of the fit (tag-to-tag x, metres). 0.080, not\n"
    + _PAD + "# 0.110: limited to the Franka Hand's 80 mm stroke (2026-11-12), so\n"
    + _PAD + "# every demonstrated width is one the robot can reproduce.\n")


def _available(ref: str) -> bool:
    if shutil.which("git") is None or not REPO.is_dir():
        return False
    try:
        _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


needs_branch = pytest.mark.skipif(not _available(BRANCH), reason=f"git or {BRANCH} not available")
needs_v2_source = pytest.mark.skipif(not _available(V2_SOURCE_COMMIT),
                                     reason=f"git or commit {V2_SOURCE_COMMIT} not available")

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


@needs_branch
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


@needs_v2_source
def test_v2_rig_is_the_users_file_except_max_opening():
    """rig_pi_v2.yaml is verbatim the user's file, except gripper.max_opening_m 0.110 -> 0.080 with its comment."""
    given = _git("show", f"{V2_SOURCE_COMMIT}:{V2_PATH}")
    old, new = V2_CHANGE
    assert given.count(old) == 1, "the stated change no longer applies"
    here = (FOLDER / "rigs" / "rig_pi_v2.yaml").read_text()
    assert here == given.replace(old, new)
    v2 = parse_rig(yaml.safe_load(here), "rig_pi_v2.yaml")
    assert v2.cameras["main"].resolution == (1600, 1200) and v2.cameras["ultrawide"].resolution == (1280, 960)
    assert v2.cameras["main"].fps == 30.0 and v2.cameras["ultrawide"].fps == 30.0
