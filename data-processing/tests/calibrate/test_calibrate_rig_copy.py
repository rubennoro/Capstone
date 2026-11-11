"""A data-root rig yaml copied before the repo rig got its ``gripper`` section: ``ensure_rig`` (calibrate, run)
stops with the section to add, instead of silently skipping the 0.110 m opening check."""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from dataproc.calibrate import calibrate_task
from dataproc.calibrate.cli import cli_calibrate
from dataproc.paths import data_root
from dataproc.pi_intake import REPO_RIGS, PiIntakeError, ensure_rig
from dataproc.raw_episode import COMPLETE, DATA_FILES, MANIFEST

RIG = "rig_pi_v1"


def _quiet(*a, **k):
    pass


@pytest.fixture
def root(tmp_path):
    return data_root(tmp_path / "root")


def _write_root_rig(root, drop_gripper: bool):
    d = yaml.safe_load((REPO_RIGS / f"{RIG}.yaml").read_text())
    assert "gripper" in d
    if drop_gripper:
        d.pop("gripper")
    root.rigs.mkdir(parents=True, exist_ok=True)
    root.rig_file(RIG).write_text(yaml.safe_dump(d, sort_keys=False))


def _minimal_pi_recording(d):
    """Passes ``inspect_source`` (COMPLETE, a matching manifest, a valid meta.json): enough to reach the rig."""
    d.mkdir(parents=True)
    for name in DATA_FILES:
        (d / name).write_bytes(b"not used")
    (d / "meta.json").write_text(json.dumps({"episode": 1, "valid": True, "problems": []}))
    names = sorted(DATA_FILES)
    (d / MANIFEST).write_text("".join(f"{hashlib.sha256((d / n).read_bytes()).hexdigest()}  {n}\n" for n in names))
    (d / COMPLETE).touch()
    return d


def test_stale_copy_is_refused(root):
    _write_root_rig(root, drop_gripper=True)
    with pytest.raises(PiIntakeError) as e:
        ensure_rig(root, RIG, _quiet)
    msg = str(e.value)
    assert "has no gripper section" in msg and "max_opening_m: 0.11" in msg and f"delete {root.rig_file(RIG)}" in msg


def test_current_copy_and_missing_copy_are_fine(root, tmp_path):
    _write_root_rig(root, drop_gripper=False)
    ensure_rig(root, RIG, _quiet)                                  # has the section: nothing to say
    other = data_root(tmp_path / "other")
    ensure_rig(other, RIG, _quiet)                                 # no copy yet: the repo one is copied
    assert other.rig_file(RIG).read_bytes() == (REPO_RIGS / f"{RIG}.yaml").read_bytes()


def test_rig_without_a_repo_copy_is_not_checked(root):
    root.rigs.mkdir(parents=True)
    root.rig_file("my_rig").write_text("rig_id: my_rig\n")         # no configs/rigs/my_rig.yaml
    ensure_rig(root, "my_rig", _quiet)


def test_calibrate_stops_on_a_stale_copy(root, tmp_path, capsys):
    _write_root_rig(root, drop_gripper=True)
    src = tmp_path / "src"
    _minimal_pi_recording(src / "episode_000001")
    with pytest.raises(PiIntakeError, match="no gripper section"):
        calibrate_task(root, "block_raise", "episode_000001", source=src, rig_id=RIG, log=_quiet)
    assert cli_calibrate(root, "block_raise", "episode_000001", source=str(src), rig_id=RIG) == 1
    err = capsys.readouterr().err
    assert "no gripper section" in err and "gripper:" in err
    assert not root.tasks.exists()
