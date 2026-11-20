# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_copies.py @ 2ffe641
# changes: checks this package's copies of rnoro/final-deployment @ 2ffe641, ea9f7fb or 927a5ba (read with `git show`,
# never changed) instead of the copies of rnoro/data-work; every frames file names its source in its first header line.
"""Each file copied into teleop_collect.frames (and the copied tests) equals its source at the rnoro/final-deployment
commit its header names (2ffe641 or the re-syncs ea9f7fb, 927a5ba), apart from the two header lines and the import
paths (STRUCTURE.md 1, PLAN rule 1), and that source is unchanged at 927a5ba; each adapted file names an existing
source.
Skipped when git or ~/Capstone-final-deployment is not available."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from frames_helpers import FOLDER

REPO = Path("~/Capstone-final-deployment").expanduser()
REFS = ("2ffe641", "ea9f7fb", "927a5ba")   # the first copy, the re-syncs (F_T_TCP, rig_pi_v3_640; the start pose)
LATEST = REFS[-1]
PREFIX = "# copied from rnoro/final-deployment:deployment/"
ADAPTED = "# adapted from rnoro/final-deployment:deployment/"
IMPORTS = [("from policy_deploy.", "from teleop_collect."), ("import policy_deploy.", "import teleop_collect.")]
FRAMES = FOLDER / "teleop_collect" / "frames"
TESTS = FOLDER / "tests" / "frames"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout


def _available() -> bool:
    if shutil.which("git") is None or not REPO.is_dir():
        return False
    try:
        for ref in REFS:
            _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason=f"git or {REPO} @ {LATEST} not available")


def _files() -> list[Path]:
    return sorted([*FRAMES.rglob("*.py"), *TESTS.glob("test_*.py")])


def _source(first_line: str, prefix: str) -> tuple[str, str]:
    """(source path under deployment/, the commit named)."""
    path, at = first_line[len(prefix):].rsplit(" @ ", 1)
    ref = at.split()[0]
    assert ref in REFS, first_line
    return path.split(",")[0].split(" (")[0], ref


COPIED = [p for p in _files() if p.read_text().startswith(PREFIX)]


def test_there_are_copies():
    assert {p.name for p in COPIED} >= {"camera_health.py", "pair_rule.py", "packet_pairer.py", "fake_pi_sender.py",
                                        "pi_recording.py", "recording_packet_source.py", "config.py"}


@pytest.mark.parametrize("path", COPIED, ids=lambda p: str(p.relative_to(FOLDER)))
def test_copy_equals_source(path):
    lines = path.read_text().split("\n")
    src, ref = _source(lines[0], PREFIX)
    assert lines[1] in ("# changes: none", "# changes: imports only"), lines[1]
    expected = _git("show", f"{ref}:deployment/{src}")
    assert expected == _git("show", f"{LATEST}:deployment/{src}"), f"{src} changed after {ref}: re-copy it"
    if lines[1] == "# changes: imports only":
        for old, new in IMPORTS:
            expected = expected.replace(old, new)
    assert "\n".join(lines[2:]) == expected


def test_the_small_test_rig_equals_the_file_on_disk():
    """rig_small.yaml is not in commit 2ffe641 (the root .gitignore ignores data/), so it is compared with the
    file on disk."""
    src = REPO / "deployment/policy_deploy/tests/frames/data/rig_small.yaml"
    here = (TESTS / "test_rigs" / "rig_small.yaml").read_text().split("\n", 2)
    assert here[0].startswith("# copied from ~/Capstone-final-deployment/") and "changes: none" in here[1]
    assert here[2] == src.read_text()


NEW = {"mkv_writer.py", "recorder.py", "preview.py", "servers.py", "test_mkv_writer.py", "test_recorder.py",
       "test_servers.py"}


@pytest.mark.parametrize("path", [p for p in _files() if p not in COPIED and p.name not in NEW],
                         ids=lambda p: str(p.relative_to(FOLDER)))
def test_adapted_files_name_an_existing_source_and_their_changes(path):
    lines = path.read_text().split("\n")
    assert lines[0].startswith(ADAPTED), lines[0]
    assert lines[1].startswith("# changes: ")
    src, ref = _source(lines[0], ADAPTED)
    _git("cat-file", "-e", f"{ref}:deployment/{src}")
