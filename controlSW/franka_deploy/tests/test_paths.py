import os
from pathlib import Path

import pytest

from franka_deploy.paths import REPO_ROOT, ForbiddenPathError, check_writable_dir, stage_checkpoint


def test_repo_root_detected():
    assert (REPO_ROOT / "behavior_prompting").is_dir() and (REPO_ROOT / "controlSW").is_dir()


@pytest.mark.parametrize("sub", ["", ".", "results", "results/block_raise/x", "data/raw", "behavior_prompting",
                                 "iPhUMI/python_package", "scripts/regression", "sensor_rig", "sensor_rig/tests",
                                 "controlSW", "controlSW/franka_deploy/work", "work_in_repo_root"])
def test_forbidden_dirs_refused(sub):
    with pytest.raises(ForbiddenPathError):
        check_writable_dir(REPO_ROOT / sub)


def test_outside_repo_allowed(tmp_path):
    assert check_writable_dir(tmp_path / "w") == tmp_path / "w"


def test_dotdot_into_repo_refused(tmp_path):
    with pytest.raises(ForbiddenPathError):
        check_writable_dir(str(REPO_ROOT / "results" / ".." / "controlSW"))


def test_symlink_into_results_refused(tmp_path):
    link = tmp_path / "sneaky"
    link.symlink_to(REPO_ROOT / "results")
    with pytest.raises(ForbiddenPathError):
        check_writable_dir(link / "sub")


def test_stage_checkpoint(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    ck = src_dir / "m.ckpt"
    ck.write_bytes(b"x")
    work = tmp_path / "work"
    link = stage_checkpoint(ck, work)
    assert link == work / "ckpt" / "m.ckpt" and link.is_symlink() and link.resolve() == ck.resolve()
    # upstream writes ckpt_path.replace(".ckpt", ".yaml"): next to the link, not next to the source
    assert Path(str(link).replace(".ckpt", ".yaml")).parent == work / "ckpt"
    assert stage_checkpoint(ck, work) == link  # idempotent
    assert stage_checkpoint(link, work) == link  # already staged
    other = src_dir / "sub"
    other.mkdir()
    ck2 = other / "m.ckpt"
    ck2.write_bytes(b"y")
    with pytest.raises(FileExistsError):
        stage_checkpoint(ck2, work)
    assert sorted(os.listdir(src_dir)) == ["m.ckpt", "sub"]  # nothing written next to the source


def test_stage_checkpoint_refuses_work_dir_in_results(tmp_path):
    ck = tmp_path / "m.ckpt"
    ck.write_bytes(b"x")
    with pytest.raises(ForbiddenPathError):
        stage_checkpoint(ck, REPO_ROOT / "results" / "block_raise")
    assert not (REPO_ROOT / "results" / "block_raise" / "ckpt").exists()


def test_stage_checkpoint_requires_ckpt(tmp_path):
    with pytest.raises(ValueError):
        stage_checkpoint(tmp_path / "m.pt", tmp_path / "w")
    with pytest.raises(FileNotFoundError):
        stage_checkpoint(tmp_path / "missing.ckpt", tmp_path / "w")
