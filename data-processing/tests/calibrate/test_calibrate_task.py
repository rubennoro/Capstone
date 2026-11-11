"""``dataproc calibrate``: sim calibration recordings in the Pi's format -> a new task folder, or nothing at all.

One ``--fast`` sim calibration episode is rendered once and written in the Pi format several times (different Pi
episode numbers, so different manifests = different recordings). Each test copies what it needs into its own
source folder and data root. Every failure must leave no task folder, an empty ``incoming/``, no
``pi_sources.csv`` row and the recording untouched (so the same command can be retried), and print its hint.
"""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from dataproc.calibrate import CalibrateError, calibrate_task
from dataproc.calibrate.cli import cli_calibrate
from dataproc.config import load_task_config
from dataproc.files import load_status
from dataproc.paths import data_root
from dataproc.pi_adapter import SOURCES_FILE, read_sources
from dataproc.process import run_process
from dataproc.raw_episode import COMPLETE
from dataproc.sim.episode import DEFAULT_RIG_PATH, fast_rig, run_sim
from dataproc.stages.calibration import GRIPPER_RANGE_FILE, calibration_inputs_hash
from pi_fixtures import contract_to_pi
from test_process_common import quiet, tree_digest

TASK = "block_raise"
RIG = "rig_v1_fast"
N_GOOD = 4


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """``{name: Pi-format recording}``: ``cal_0..cal_3`` good, ``cal_invalid`` marked invalid by the Pi."""
    base = tmp_path_factory.mktemp("calibrate_src")
    sim = base / "sim"
    run_sim(sim, episodes=1, calibration=True, fast=True, session_id="cal", log=quiet)
    (ep,) = sorted(sim.iterdir())
    pi = base / "pi"
    out = {f"cal_{k}": contract_to_pi(ep, pi / f"cal_{k}", k, with_exposure=True) for k in range(N_GOOD)}
    out["cal_invalid"] = contract_to_pi(ep, pi / "cal_invalid", 90, with_exposure=True, valid=False)
    return out


def _rig_dict(**changes) -> dict:
    d, _ = fast_rig(DEFAULT_RIG_PATH)
    for path, value in changes.items():
        node = d
        *parents, key = path.split(".")
        for p in parents:
            node = node[p]
        if value is None:
            node.pop(key)
        else:
            node[key] = value
    return d


@pytest.fixture
def env(tmp_path, pristine):
    """``(root, source, add)``: an empty data root with ``rigs/rig_v1_fast.yaml``; ``add(name)`` copies a pristine
    recording into the source folder; ``rig(rig_id, **changes)`` writes a modified fast rig."""
    root = data_root(tmp_path / "root")
    source = tmp_path / "pi_incoming"
    source.mkdir()

    def add(name: str, as_name: str | None = None):
        return shutil.copytree(pristine[name], source / (as_name or name))

    def rig(rig_id: str, **changes) -> str:
        d = _rig_dict(**changes)
        d["rig_id"] = rig_id
        root.rigs.mkdir(parents=True, exist_ok=True)
        (root.rigs / f"{rig_id}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
        return rig_id

    rig(RIG)
    return root, source, add, rig


def _assert_nothing_taken(root, src_dir, before):
    assert not root.tasks.exists() or list(root.tasks.iterdir()) == []
    assert not root.incoming.exists() or list(root.incoming.iterdir()) == []
    assert read_sources(root.path / SOURCES_FILE) == []
    if before is not None:
        assert tree_digest(src_dir) == before


def _fail(capsys, root, source, name, rig=RIG, expect_hint=(), task=TASK):
    src_dir = source / name
    before = tree_digest(src_dir) if src_dir.is_dir() else None
    assert cli_calibrate(root, task, name, source=source, rig_id=rig) == 1
    err = capsys.readouterr().err
    assert f"calibrate: FAILED for {name}" in err
    for h in expect_hint:
        assert h in err, err
    assert "nothing was created" in err
    _assert_nothing_taken(root, src_dir, before)
    return err


# --------------------------------------------------------------------------------------------- success

def test_folders_are_task_then_task_2_then_task_3(env, capsys):
    root, source, add, _ = env
    for k in range(3):
        add(f"cal_{k}")
    expect = [TASK, f"{TASK}_2", f"{TASK}_3"]
    for k, folder in enumerate(expect):
        assert cli_calibrate(root, TASK, f"cal_{k}", source=source, rig_id=RIG) == 0
        out = capsys.readouterr().out
        assert f"calibrate: OK -> tasks/{folder} (label {TASK}" in out
        assert "measured opening 0.06" in out and "expected 0.060 m ± 0.010 m" in out
        assert f"dataproc run {folder} <start_dir> <end_dir> --source {source}" in out
        assert "warning: only 1 open/close cycle(s)" in out          # the 5 s sim sweep opens once
    assert sorted(p.name for p in root.tasks.iterdir()) == sorted(expect)
    for folder in expect:
        td = root.tasks / folder
        cfg = load_task_config(td / "task.yaml")
        assert (cfg.label, cfg.rig) == (TASK, RIG)
        (cal,) = (td / "calibration").iterdir()
        rec = load_status(cal / "processed")["gripper_range"]
        assert rec["status"] == "done" and rec["reasons"] == []
        assert rec["inputs_hash"] == calibration_inputs_hash(cal / "raw", fast_rig(DEFAULT_RIG_PATH)[1], RIG)
        assert (td / GRIPPER_RANGE_FILE).read_text() == (cal / "processed" / GRIPPER_RANGE_FILE).read_text()
    rows = read_sources(root.path / SOURCES_FILE)
    assert [(r["source_name"], r["task"], r["kind"], r["status"]) for r in rows] == [
        (f"cal_{k}", f, "calibration", "ingested") for k, f in enumerate(expect)]
    assert list(root.incoming.iterdir()) == []


def test_process_reuses_the_calibration_without_fitting(env):
    root, source, add, _ = env
    src = add("cal_0")
    before = tree_digest(src)
    res = calibrate_task(root, TASK, "cal_0", source=source, rig_id=RIG, log=quiet)
    assert res.ok and res.folder == TASK and res.slot.startswith("calibration/")
    assert abs(res.fit.opening_m - 0.060) < 0.002
    cal = res.task_dir / res.slot
    rec = load_status(cal / "processed")["gripper_range"]
    pr = run_process(root, TASK, slam="oracle", log=quiet)
    assert pr.report["calibration"]["state"] == "done"
    assert pr.report["calibration"]["ran"] is False                    # reused: not fitted again
    assert load_status(cal / "processed")["gripper_range"] == rec      # record untouched
    assert tree_digest(src) == before                                  # the Pi's folder is never touched


def test_rig_without_gripper_section_skips_the_opening_check(env):
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_nogripper", gripper=None)
    res = calibrate_task(root, TASK, "cal_0", source=source, rig_id=rid, log=quiet)
    assert res.ok
    assert any("no gripper section" in w and "opening check skipped" in w for w in res.warnings)


# --------------------------------------------------------------------------------------------- failures

def test_missing_directory(env, capsys):
    root, source, _, _ = env
    _fail(capsys, root, source, "cal_nope", expect_hint=["must be a recording directly inside --source"])


def test_still_syncing_then_retry(env, capsys):
    root, source, add, _ = env
    src = add("cal_0")
    (src / COMPLETE).unlink()
    _fail(capsys, root, source, "cal_0", expect_hint=["no COMPLETE", "still being recorded or transferred"])
    (src / COMPLETE).touch()                                           # transfer finished: same command works
    assert cli_calibrate(root, TASK, "cal_0", source=source, rig_id=RIG) == 0


def test_manifest_mismatch(env, capsys):
    root, source, add, _ = env
    src = add("cal_0")
    with open(src / "imu.csv", "a") as f:
        f.write("\n")
    _fail(capsys, root, source, "cal_0", expect_hint=["manifest check failed", "re-send the recording"])


def test_marked_invalid_by_the_pi(env, capsys):
    root, source, add, _ = env
    add("cal_invalid")
    _fail(capsys, root, source, "cal_invalid", expect_hint=["marked it invalid", "record a new calibration"])


def test_already_taken(env, capsys):
    root, source, add, _ = env
    add("cal_0")
    assert cli_calibrate(root, TASK, "cal_0", source=source, rig_id=RIG) == 0
    capsys.readouterr()
    assert cli_calibrate(root, "other_task", "cal_0", source=source, rig_id=RIG) == 1
    err = capsys.readouterr().err
    assert "already taken (calibration of task folder block_raise, ingested)" in err
    assert "taken in once" in err
    assert sorted(p.name for p in root.tasks.iterdir()) == [TASK]
    assert len(read_sources(root.path / SOURCES_FILE)) == 1 and list(root.incoming.iterdir()) == []


def test_rig_mismatch_then_retry_with_the_right_rig(env, capsys):
    root, source, add, _ = env
    add("cal_0")
    err = _fail(capsys, root, source, "cal_0", rig="rig_pi_v1", expect_hint=["wrong --rig rig_pi_v1"])
    assert "ultrawide camera resolution" in err
    assert cli_calibrate(root, TASK, "cal_0", source=source, rig_id=RIG) == 0


def test_tags_never_seen(env, capsys):
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_other_ids", **{"aruco.left_id": 10, "aruco.right_id": 11})
    err = _fail(capsys, root, source, "cal_0", rig=rid,
                expect_hint=["never both detected", "were never seen", "aruco.dict (DICT_4X4_50)"])
    assert "tag 10 seen 0 (0%), tag 11 seen 0 (0%)" in err


def test_one_tag_never_seen(env, capsys):
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_one_id", **{"aruco.right_id": 11})
    _fail(capsys, root, source, "cal_0", rig=rid, expect_hint=["tag 11 (right finger) was never seen"])


def test_tags_outside_the_depth_window(env, capsys):
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_far", **{"aruco.nominal_z_m": 0.5})
    err = _fail(capsys, root, source, "cal_0", rig=rid,
                expect_hint=["not enough valid both-tag frames", "fix aruco.nominal_z_m", "remounted"])
    assert "both tags at the expected depth (z 0.420-0.580 m): 0; median z 0.09" in err


def test_opening_out_of_tolerance(env, capsys):
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_wide", **{"gripper.max_opening_m": 0.110})
    err = _fail(capsys, root, source, "cal_0", rig=rid,
                expect_hint=["measured opening 0.060 m, expected 0.110 m ± 0.010 m", "aruco.marker_size_m (0.016 m)",
                             "measured / expected = 0.55", "not opened fully"])
    assert "open/close cycles: 1; closed 0.030 m, open 0.090 m" in err
    # the tolerance is honoured on both sides
    rid = rig("rig_edge", **{"gripper.max_opening_m": 0.069, "gripper.opening_tolerance_m": 0.010})
    assert calibrate_task(root, TASK, "cal_0", source=source, rig_id=rid, log=quiet).ok


def test_bad_arguments_raise(env):
    root, source, _, _ = env
    with pytest.raises(CalibrateError, match="invalid task name"):
        calibrate_task(root, "bad/name", "cal_0", source=source, rig_id=RIG, log=quiet)
    with pytest.raises(CalibrateError, match="not a directory"):
        calibrate_task(root, TASK, "cal_0", source=source / "nope", rig_id=RIG, log=quiet)
    with pytest.raises(CalibrateError, match="invalid episode directory name"):
        calibrate_task(root, TASK, "../x", source=source, rig_id=RIG, log=quiet)
    assert not root.tasks.exists()


def test_stray_incoming_entry_stops_before_anything(env, capsys):
    root, source, add, _ = env
    add("cal_0")
    (root.incoming / "someone_elses").mkdir(parents=True)
    assert cli_calibrate(root, TASK, "cal_0", source=source, rig_id=RIG) == 1
    assert "is not empty (someone_elses)" in capsys.readouterr().err
    assert not root.tasks.exists() and read_sources(root.path / SOURCES_FILE) == []


def test_failed_fit_record_is_not_written_anywhere(env):
    """A failed calibrate leaves no processed/ or status anywhere: the next attempt starts clean."""
    root, source, add, rig = env
    add("cal_0")
    rid = rig("rig_wide2", **{"gripper.max_opening_m": 0.110})
    res = calibrate_task(root, TASK, "cal_0", source=source, rig_id=rid, log=quiet)
    assert not res.ok and res.reasons == ["opening_out_of_range"] and res.folder is None
    assert json.loads(json.dumps(res.fit.diagnostics))["n_in_window"] > 0
    _assert_nothing_taken(root, source / "cal_0", None)


def test_process_refits_when_the_gripper_section_changes(env):
    """The gripper section is in the calibration inputs hash: editing it makes process fit (and check) again."""
    root, source, add, rig = env
    add("cal_0")
    assert calibrate_task(root, TASK, "cal_0", source=source, rig_id=RIG, log=quiet).ok
    rig(RIG, **{"gripper.max_opening_m": 0.110})
    pr = run_process(root, TASK, slam="oracle", log=quiet)
    assert pr.report["calibration"]["ran"] is True
    assert pr.report["calibration"]["state"] == "calibration_failed"
    assert "measured opening 0.060 m, expected 0.110 m" in pr.report["calibration"]["reason"]
    assert not (root.tasks / TASK / GRIPPER_RANGE_FILE).exists()
