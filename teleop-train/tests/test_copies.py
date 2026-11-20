"""Each file in teleop_train/_copied/ equals its source at the recorded commit (PLAN.md §3 rule 3, §11).

The copy is: one ``# copied from <branch>:<path> @ <commit>`` line, an optional ``# changes: ...`` line, then the
source with only the listed import replacements. Skipped when the source worktree is not on this machine.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

COPIED = Path(__file__).resolve().parents[1] / "teleop_train" / "_copied"
DATA_WORK = Path("~/Capstone-data-work").expanduser()
FINAL_DEPLOYMENT = Path("~/Capstone-final-deployment").expanduser()
UP = "umi-processing/umiproc/dp"
FD = "deployment/policy_deploy/policy_deploy/frames/_dataproc"

# (file under _copied/, worktree, commit, branch, source path, [(old, new) import replacements])
CASES = [
    ("frames.py", DATA_WORK, "839d23b", "rnoro/data-work-auto", f"{UP}/export/frames.py", []),
    ("zarr_writer.py", DATA_WORK, "839d23b", "rnoro/data-work-auto", f"{UP}/export/zarr_writer.py",
     [("from .._vendor.", "from ._vendor.")]),
    ("_vendor/__init__.py", DATA_WORK, "839d23b", "rnoro/data-work-auto", f"{UP}/_vendor/__init__.py", []),
    ("_vendor/replay_buffer.py", DATA_WORK, "839d23b", "rnoro/data-work-auto", f"{UP}/_vendor/replay_buffer.py", []),
    ("_vendor/imagecodecs_numcodecs.py", DATA_WORK, "839d23b", "rnoro/data-work-auto",
     f"{UP}/_vendor/imagecodecs_numcodecs.py", []),
] + [
    (f"_dataproc/{f}", FINAL_DEPLOYMENT, "2ffe641", "rnoro/final-deployment", f"{FD}/{f}", [])
    for f in ("__init__.py", "config.py", "paths.py", "imageprep.py", "geometry/__init__.py",
              "geometry/camera_model.py", "geometry/transforms.py")
]


def _source(worktree: Path, commit: str, path: str) -> str | None:
    if not worktree.is_dir():
        return None
    r = subprocess.run(["git", "-C", str(worktree), "show", f"{commit}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


@pytest.mark.parametrize("rel, worktree, commit, branch, src_path, replacements", CASES, ids=[c[0] for c in CASES])
def test_copy_matches_source(rel, worktree, commit, branch, src_path, replacements):
    src = _source(worktree, commit, src_path)
    if src is None:
        pytest.skip(f"{worktree} @ {commit} not available")
    lines = (COPIED / rel).read_text(encoding="utf-8").splitlines(keepends=True)
    assert lines[0] == f"# copied from {branch}:{src_path} @ {commit}\n"
    body = lines[1:]
    if replacements:
        assert body[0].startswith("# changes: imports only"), "a copy with import changes must say so"
        body = body[1:]
    else:
        assert not body[0].startswith("# changes:"), "an unchanged copy has no changes line"
    expected = src
    for old, new in replacements:
        assert old in expected, f"{old!r} not in the source any more"
        expected = expected.replace(old, new)
    assert "".join(body) == expected


def test_every_copied_file_is_checked():
    files = {str(p.relative_to(COPIED)) for p in COPIED.rglob("*.py")} - {"__init__.py"}
    assert files == {c[0] for c in CASES}
