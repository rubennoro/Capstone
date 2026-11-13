"""Each COPIED file equals its source on rnoro/data-work, apart from the provenance header and the listed changes,
and the rig file rig_pi_v2.yaml equals the user's file as given apart from its two stated changes (STRUCTURE.md §3).
Skipped when git or the source is not available."""

from __future__ import annotations

import re
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


#: The commit that added rig_pi_v2.yaml exactly as the user gave it, and the two changes made to it since.
V2_SOURCE_COMMIT = "196eae4"
V2_PATH = "deployment/process_frames/rigs/rig_pi_v2.yaml"
_PAD = " " * 54
V2_CHANGE = (
    "  max_opening_m: 0.110" + " " * 32 + "# max_width - min_width of the fit (tag-to-tag x, metres)\n",
    "  max_opening_m: 0.080" + " " * 32 + "# max_width - min_width of the fit (tag-to-tag x, metres). 0.080, not\n"
    + _PAD + "# 0.110: limited to the Franka Hand's 80 mm stroke (2026-11-12), so\n"
    + _PAD + "# every demonstrated width is one the robot can reproduce.\n")
#: The tcp block as given (nominal CAD); the file now holds the one `python -m deploy.geometry` wrote instead.
V2_GIVEN_TCP = (
    "tcp:" + " " * 50 + "# T_cam_tcp is in the PHYSICAL ultrawide optical frame\n"
    "  T_cam_tcp:" + " " * 42 + "# nominal CAD, as rig_v1\n"
    "    rotation: {euler_xyz_deg: [15.0, 0.0, 0.0]}\n"
    "    translation_m: [0.0, 0.060, 0.200]\n")
V2_TCP_LINES = (r"# T_cam_tcp COMPUTED by `python -m deploy\.geometry` on \d{4}-\d{2}-\d{2} from \S+/gripper\.yaml",
                r"# and franka_control tcp\.F_T_TCP\. Do not edit by hand: change gripper\.yaml and rerun the tool\.",
                r"tcp:",
                r"  T_cam_tcp:",
                r"    matrix: \[.*\]   # 4x4, row-major")
#: What deploy.geometry computes from deployment/gripper.yaml (ultrawide at (55, 0, 10) mm on the hand's +x side,
#: tilted 15 deg) and franka_control's F_T_TCP (-45 deg about the flange z, fingertips 150 mm out). deploy's
#: test_geometry checks that these files reproduce the rig file's matrix.
_C, _S = np.cos(np.radians(15.0)), np.sin(np.radians(15.0))
V2_T_CAM_TCP = np.array([[0.0, 1.0, 0.0, 0.0],
                         [-_C, 0.0, -_S, 0.016891254131545838],
                         [-_S, 0.0, _C, 0.1494646631611082],
                         [0.0, 0.0, 0.0, 1.0]])


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
    assert NAME_RE.fullmatch("rig_pi_v2") and not NAME_RE.fullmatch("a b")


@needs_v2_source
def test_v2_rig_is_the_users_file_except_max_opening_and_tcp():
    """rig_pi_v2.yaml is verbatim the user's file, except (1) gripper.max_opening_m 0.110 -> 0.080 with its comment and
    (2) the tcp block, which holds the T_cam_tcp matrix deploy.geometry computes (with its COMPUTED header)."""
    given = _git("show", f"{V2_SOURCE_COMMIT}:{V2_PATH}")
    old, new = V2_CHANGE
    assert given.count(old) == 1 and given.count(V2_GIVEN_TCP) == 1, "a stated change no longer applies"
    here = (FOLDER / "rigs" / "rig_pi_v2.yaml").read_text()
    lines = here.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("# T_cam_tcp COMPUTED"))
    tcp_lines = lines[start:start + len(V2_TCP_LINES)]
    for line, pattern in zip(tcp_lines, V2_TCP_LINES, strict=True):
        assert re.fullmatch(pattern, line.rstrip("\n")), line
    rest = "".join(lines[:start] + [V2_GIVEN_TCP] + lines[start + len(V2_TCP_LINES):])
    assert rest == given.replace(old, new)
    v2 = parse_rig(yaml.safe_load(here), "rig_pi_v2.yaml")
    assert np.allclose(v2.T_cam_tcp, V2_T_CAM_TCP, atol=1e-9, rtol=0.0)
    assert v2.cameras["main"].resolution == (1600, 1200) and v2.cameras["ultrawide"].resolution == (1280, 960)
    assert v2.cameras["main"].fps == 30.0 and v2.cameras["ultrawide"].fps == 30.0
    assert v2.gripper.max_opening_m == 0.080
