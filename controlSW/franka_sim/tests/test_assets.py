"""Vendored menagerie assets: SOURCE.md matches the tree, the models compile, tampering is detected."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from franka_sim.tools import check_assets

ASSETS = check_assets.DEFAULT_ASSETS


def test_check_assets_passes_on_committed_tree():
    n, problems = check_assets.check(ASSETS)
    assert problems == []
    assert n >= 100
    text = (ASSETS / "SOURCE.md").read_text()
    sha = [ln for ln in text.splitlines() if ln.startswith("- commit:")][0].split(":", 1)[1].strip()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def _hinges(m):
    import mujoco

    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]


@pytest.mark.parametrize("xml,expected", [
    ("franka_emika_panda/panda.xml", [f"joint{i}" for i in range(1, 8)]),
    ("franka_fr3/fr3.xml", [f"fr3_joint{i}" for i in range(1, 8)]),
])
def test_arm_models_compile(xml, expected):
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(ASSETS / xml))
    assert _hinges(m) == expected


def test_hand_compiles_standalone():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "franka_emika_panda/hand.xml"))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert names == ["finger_joint1", "finger_joint2"]


def test_tampered_copy_fails(tmp_path: Path):
    copy = tmp_path / "assets"
    copy.mkdir()
    shutil.copyfile(ASSETS / "SOURCE.md", copy / "SOURCE.md")
    target = copy / "franka_fr3" / "fr3.xml"
    target.parent.mkdir()
    shutil.copyfile(ASSETS / "franka_fr3" / "fr3.xml", target)
    # only fr3.xml is present: everything else is reported missing, but fr3.xml itself must be clean
    _, problems = check_assets.check(copy)
    assert "sha256 mismatch: franka_fr3/fr3.xml" not in problems
    data = bytearray(target.read_bytes())
    data[0] ^= 0x01
    target.write_bytes(bytes(data))
    n, problems = check_assets.check(copy)
    assert "sha256 mismatch: franka_fr3/fr3.xml" in problems
    assert check_assets.run(type("A", (), {"assets": str(copy)})()) == 1


def test_extra_file_is_an_error(tmp_path: Path):
    copy = tmp_path / "assets"
    shutil.copytree(ASSETS / "franka_fr3", copy / "franka_fr3")
    shutil.copyfile(ASSETS / "SOURCE.md", copy / "SOURCE.md")
    (copy / "franka_fr3" / "stray.txt").write_text("x")
    _, problems = check_assets.check(copy)
    assert "extra file not in SOURCE.md: franka_fr3/stray.txt" in problems
