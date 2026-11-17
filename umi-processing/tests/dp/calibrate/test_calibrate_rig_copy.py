"""A data-root rig yaml copied before the repo rig got its ``gripper`` section: ``ensure_rig`` (calibrate, run)
stops with the section to add, instead of silently skipping the 0.110 m opening check. The same for
``cameras.ultrawide.slam_mask`` (without it ORB-SLAM3 tracks the gripper); a copy whose mask differs from the
repo's only gets a warning (processing uses the copy)."""

from __future__ import annotations

import copy
import hashlib
import json
import textwrap

import pytest
import yaml

from umiproc.dp import pi_intake
from umiproc.dp.calibrate import calibrate_task
from umiproc.dp.calibrate.cli import cli_calibrate
from umiproc.dp.config import load_rig
from umiproc.dp.paths import data_root
from umiproc.dp.pi_intake import REPO_RIGS, PiIntakeError, check_rig_copy, ensure_rig
from umiproc.dp.raw_episode import COMPLETE, DATA_FILES, MANIFEST

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


# --------------------------------------------------------------------------------------------- slam_mask

MASK_RIG = "rig_pi_v3"                                            # ultrawide 1280x960
MASK = {"source": "traced from episode_000115", "polygons": [
    [[176, 960], [178, 905], [380, 624], [580, 462], [671, 462], [671, 960]],
    [[671, 462], [762, 462], [1280, 936], [1280, 960], [671, 960]]]}


def _mask_rig(path, mask=None):
    """The repo's rig_pi_v3 written to ``path`` with ``mask`` as its cameras.ultrawide.slam_mask (None: no mask)."""
    d = yaml.safe_load((REPO_RIGS / f"{MASK_RIG}.yaml").read_text())
    d["cameras"]["ultrawide"].pop("slam_mask", None)
    if mask is not None:
        d["cameras"]["ultrawide"]["slam_mask"] = copy.deepcopy(mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo rigs dir for ensure_rig (configs/rigs has no masked rig to test with)."""
    monkeypatch.setattr(pi_intake, "REPO_RIGS", tmp_path / "repo")
    return tmp_path / "repo"


def test_copy_without_the_repo_mask_is_refused(root, repo):
    src, dst = _mask_rig(repo / f"{MASK_RIG}.yaml", MASK), _mask_rig(root.rig_file(MASK_RIG))
    before = dst.read_bytes()
    with pytest.raises(PiIntakeError) as e:
        ensure_rig(root, MASK_RIG, _quiet)
    msg = str(e.value)
    assert msg.startswith(f"{dst} has no cameras.ultrawide.slam_mask, but the repo's {src} has one: the copy "
                          "predates it") and "ORB-SLAM3 tracks the gripper" in msg
    assert msg.endswith(f"or delete {dst} (the repo copy is then used).") and dst.read_bytes() == before
    snippet = textwrap.dedent("\n".join(ln for ln in msg.splitlines() if ln.startswith("    ")))
    assert yaml.safe_load(snippet) == {"slam_mask": MASK}
    # pasted under cameras.ultrawide, the copy is accepted and has the repo's mask
    d = yaml.safe_load(dst.read_text())
    d["cameras"]["ultrawide"].update(yaml.safe_load(snippet))
    dst.write_text(yaml.safe_dump(d, sort_keys=False))
    ensure_rig(root, MASK_RIG, _quiet)
    assert load_rig(dst).cameras["ultrawide"].slam_mask == load_rig(src).cameras["ultrawide"].slam_mask


def test_different_masks_only_warn(root, repo):
    src = _mask_rig(repo / f"{MASK_RIG}.yaml", MASK)
    other = copy.deepcopy(MASK)
    other["polygons"][1][2] = [1280, 930]
    dst = _mask_rig(root.rig_file(MASK_RIG), other)
    (w,) = check_rig_copy(dst, src)
    assert f"slam_mask polygons of {dst} differ from the repo's {src}: processing uses {dst}" in w
    logged = []
    ensure_rig(root, MASK_RIG, logged.append)
    assert logged == [f"rig: warning: {w}"]
    # the same polygons (as floats) with another note: nothing to say
    same = {"source": "another note", "polygons": [[[float(x), float(y)] for x, y in p] for p in MASK["polygons"]]}
    _mask_rig(dst, same)
    assert check_rig_copy(dst, src) == []
    ensure_rig(root, MASK_RIG, logged.append)
    assert len(logged) == 1


def test_mask_only_in_the_copy_or_nowhere_is_fine(root, repo):
    src, dst = _mask_rig(repo / f"{MASK_RIG}.yaml"), _mask_rig(root.rig_file(MASK_RIG), MASK)
    assert check_rig_copy(dst, src) == []
    _mask_rig(dst)
    assert check_rig_copy(dst, src) == []
    logged = []
    ensure_rig(root, MASK_RIG, logged.append)
    assert logged == []


def test_gripper_is_checked_before_the_mask(root, repo):
    src, dst = _mask_rig(repo / f"{MASK_RIG}.yaml", MASK), _mask_rig(root.rig_file(MASK_RIG))
    d = yaml.safe_load(dst.read_text())
    del d["gripper"]
    dst.write_text(yaml.safe_dump(d, sort_keys=False))
    with pytest.raises(PiIntakeError, match="has no gripper section"):
        check_rig_copy(dst, src)
