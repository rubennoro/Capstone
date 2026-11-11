"""``dataproc run TASK_FOLDER START_DIR END_DIR`` end to end on sim recordings in the Pi format.

The main flow is run once (module fixture ``done``) on a calibrated folder: first with ``--slam none`` (every
recording waits in ``.staging/``), then the same range with the oracle standing in for ORB-SLAM3. Demo 2 has no
``truth.npz``, so it fails at slam and goes to ``failed/``; demos 1, 3 and 4 become ``episode_1..3``. Tests that
change anything work on a copy of the root.
"""

from __future__ import annotations

import json

import pytest

from dataproc.files import load_status
from dataproc.intake import read_ledger
from dataproc.pi_adapter import SOURCES_FILE, append_sources, read_sources
from dataproc.process import run_process
from dataproc.report import STAGE_NAMES
from dataproc.run import EXIT_OK, EXIT_WAITING_FOR_SLAM, RunError, run_range
from dataproc.run.cli import cli_run
from dataproc.run.plan import RESUME, RETRY, SKIP, TAKE, natural_key, plan_run, select_range
from run_common import (RIG, TASK, calibrated_root, copy_root, pi_source, pristine, quiet, standard_source,
                        tree_digest)

GOOD = ["episode_000001", "episode_000003", "episode_000004"]
BAD = "episode_000002"


def _run(root, src, start="episode_000001", end="episode_000004", **kw):
    kw.setdefault("slam", "oracle")
    return run_range(root, TASK, start, end, source=src, log=quiet, **kw)


def _outcomes(res) -> dict:
    return {o.name: o for o in res.outcomes}


def _pi_name(res, name):
    return _outcomes(res)[name].pi_name


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    """``(root, src, res_none, res_oracle)`` after the two runs of the main flow (read-only)."""
    root0, src = calibrated_root(tmp_path_factory)
    root = copy_root(root0, tmp_path_factory.mktemp("run_done") / "root")
    res_none = _run(root, src, slam="none")
    snapshot = {"staging": sorted(p.name for p in (root.tasks / TASK / ".staging").iterdir()),
                "episodes": sorted(p.name for p in (root.tasks / TASK).glob("episode_*")),
                "sources": read_sources(root.path / SOURCES_FILE)}
    res_oracle = _run(root, src)
    return root, src, res_none, res_oracle, snapshot


@pytest.fixture
def work(done, tmp_path):
    """A writable copy of the root after the main flow, and the source."""
    root, src = done[0], done[1]
    return copy_root(root, tmp_path / "root"), src


# --------------------------------------------------------------------------------------------- arguments

def test_missing_folder(tmp_path_factory, tmp_path, capsys):
    root, src = calibrated_root(tmp_path_factory)
    with pytest.raises(RunError, match=r"task folder nope does not exist \(or has no successful calibration\): "
                                       r"run `dataproc calibrate <task> <episode_dir>` first"):
        run_range(root, "nope", "episode_000001", "episode_000002", source=src, slam="oracle", log=quiet)
    assert cli_run(root, "nope", "episode_000001", "episode_000002", source=src, slam="oracle") == 1
    assert "dataproc calibrate" in capsys.readouterr().err


def test_uncalibrated_folder(tmp_path_factory, tmp_path):
    root = copy_root(calibrated_root(tmp_path_factory)[0], tmp_path / "root")
    src = calibrated_root(tmp_path_factory)[1]
    (status,) = (root.tasks / TASK / "calibration").glob("*/processed/status.json")
    rec = json.loads(status.read_text())
    rec["gripper_range"]["status"] = "failed"
    status.write_text(json.dumps(rec))
    before = tree_digest(root.path)
    with pytest.raises(RunError, match="no successful calibration"):
        _run(root, src)
    assert tree_digest(root.path) == before
    status.unlink()
    with pytest.raises(RunError, match="no successful calibration"):
        _run(root, src)
    assert not (root.tasks / TASK / ".staging").exists()


def test_start_and_end_checks(tmp_path_factory):
    root, src = calibrated_root(tmp_path_factory)
    with pytest.raises(RunError, match="START_DIR episode_000009 is not a directory"):
        _run(root, src, start="episode_000009")
    with pytest.raises(RunError, match="END_DIR nope is not a directory"):
        _run(root, src, end="nope")
    with pytest.raises(RunError, match="comes after END_DIR"):
        _run(root, src, start="episode_000004", end="episode_000001")
    with pytest.raises(RunError, match="START_DIR"):
        _run(root, src, start="../x")


def test_range_is_inclusive_in_natural_order(tmp_path):
    for n in ("ep_1", "ep_2", "ep_9", "ep_10", "ep_11", ".ep_5", "ep_100"):
        (tmp_path / n).mkdir()
    (tmp_path / "ep_3").write_text("a file, not a recording")
    assert select_range(tmp_path, "ep_2", "ep_10") == ["ep_2", "ep_9", "ep_10"]
    assert select_range(tmp_path, "ep_9", "ep_9") == ["ep_9"]
    assert select_range(tmp_path, "ep_1", "ep_100") == ["ep_1", "ep_2", "ep_9", "ep_10", "ep_11", "ep_100"]
    assert sorted(["b10", "b9", "a"], key=natural_key) == ["a", "b9", "b10"]


# --------------------------------------------------------------------------------------------- skip reasons

def test_every_skip_reason(tmp_path_factory, tmp_path, capsys):
    """Dry run on a source with one recording of each kind (nothing is changed, nothing is taken)."""
    root = copy_root(calibrated_root(tmp_path_factory)[0], tmp_path / "root")
    p = pristine(tmp_path_factory)
    src = tmp_path / "src"
    pi_source(p["cal"], src / "r_0", 0)                                   # == the folder's calibration
    pi_source(p["demos"][0], src / "r_1", 11, complete=False)             # still syncing
    bad = pi_source(p["demos"][0], src / "r_2", 12)                       # manifest failed
    (bad / "imu.csv").write_bytes((bad / "imu.csv").read_bytes() + b"\n")
    pi_source(p["demos"][0], src / "r_3", 13, valid=False)               # marked invalid by the Pi
    other = pi_source(p["demos"][0], src / "r_4", 14)                     # taken by another task folder
    from dataproc.fsutil import sha256_file
    append_sources(root.path / SOURCES_FILE, [{"source_name": "r_4", "source_manifest_sha256":
                                               sha256_file(other / "MANIFEST.sha256"), "converted_name": "x",
                                               "task": "cup_push", "kind": "demonstration", "status": "ingested"}])
    pi_source(p["demos"][1], src / "r_5", 15)                             # new: taken
    (src / "r_cal_copy").mkdir()                                          # not a Pi recording at all
    before = tree_digest(root.path)
    res = run_range(root, TASK, "r_0", "r_5", source=src, slam="oracle", dry_run=True, log=quiet)
    plan = {r.name: r for r in res.plan.recordings}
    assert [r.name for r in res.plan.recordings] == ["r_0", "r_1", "r_2", "r_3", "r_4", "r_5"]
    assert plan["r_0"].why.startswith("is a calibration (of task folder block_raise")
    assert plan["r_1"].why.startswith("still syncing")
    assert plan["r_2"].why.startswith("manifest check failed")
    assert "the Pi marked it invalid" in plan["r_3"].why
    assert plan["r_4"].why.startswith("taken by another task folder (cup_push")
    assert plan["r_5"].action == TAKE and all(plan[n].action == SKIP for n in ("r_0", "r_1", "r_2", "r_3", "r_4"))
    assert tree_digest(root.path) == before
    assert cli_run(root, TASK, "r_0", "r_5", source=str(src), slam="oracle", dry_run=True) == 0
    out = capsys.readouterr().out
    assert "r_5: take" in out and "r_1: skip (still syncing" in out and "nothing was changed" in out
    assert tree_digest(root.path) == before


# --------------------------------------------------------------------------------------------- the main flow

def test_without_slam_everything_waits(done):
    root, src, res, _, snap = done
    assert res.exit_code == EXIT_WAITING_FOR_SLAM and res.error is None
    oc = _outcomes(res)
    assert [oc[n].state for n in GOOD + [BAD]] == ["waiting"] * 4
    assert all(oc[n].ran == ["check", "slam"] for n in GOOD + [BAD])
    assert snap["staging"] == sorted(oc[n].pi_name for n in GOOD + [BAD]) and snap["episodes"] == []
    assert sorted((r["source_name"], r["kind"], r["status"]) for r in snap["sources"]) == [
        ("episode_000000", "calibration", "ingested")] + [(n, "demonstration", "ingested") for n in sorted(GOOD + [BAD])]


def test_resume_with_slam_gives_gap_free_episodes(done):
    root, src, _, res, _ = done
    td = root.tasks / TASK
    assert res.exit_code == EXIT_OK and res.error is None
    oc = _outcomes(res)
    assert [oc[n].slot for n in GOOD] == ["episode_1", "episode_2", "episode_3"]
    for n in GOOD:
        assert oc[n].state == "episode" and "check" not in oc[n].ran and oc[n].ran[0] == "slam"
        ep = td / oc[n].slot
        assert (ep / "processed" / "episode.zarr").is_dir()
        assert set(load_status(ep / "processed")) == set(STAGE_NAMES)
    assert sorted(p.name for p in td.glob("episode_*")) == ["episode_1", "episode_2", "episode_3"]
    assert not any((td / ".staging").iterdir()) and not any(root.incoming.iterdir())


def test_failing_recording_goes_to_failed(done):
    root, src, _, res, _ = done
    td = root.tasks / TASK
    o = _outcomes(res)[BAD]
    assert o.state == "failed" and o.stage == "slam" and o.reasons == ["oracle_no_truth"]
    assert o.describe() == "failed (slam: oracle_no_truth)"
    rec = json.loads((td / "failed" / o.pi_name / "reason.json").read_text())
    assert rec["stage"] == "slam" and rec["status"] == "failed" and rec["reasons"] == ["oracle_no_truth"]
    assert "truth.npz" in rec["message"] and rec["time"] and rec["source_name"] == BAD
    assert (td / "failed" / o.pi_name / "raw").is_dir()
    rows = {r["source_name"]: r["status"] for r in read_sources(root.path / SOURCES_FILE)}
    assert rows[BAD] == "failed" and all(rows[n] == "ingested" for n in GOOD)


def test_ledger_follows_the_moves(done):
    """Each slot's newest ledger row names the recording that is there; episode.zarr carries the same name."""
    root, _, _, res, _ = done
    td = root.tasks / TASK
    newest = {}
    for r in read_ledger(td / "episodes.csv"):
        newest[r["slot"]] = r["pi_name"]
    for n in GOOD:
        o = _outcomes(res)[n]
        assert newest[o.slot] == o.pi_name
        attrs = json.loads((td / o.slot / "processed" / "episode.zarr" / ".zattrs").read_text())
        assert attrs["name"] == o.pi_name and attrs["task_label"] == TASK
    assert newest[f"failed/{_pi_name(res, BAD)}"] == _pi_name(res, BAD)


def test_report_json(done):
    root, _, _, res, _ = done
    rep = json.loads((root.tasks / TASK / "report.json").read_text())
    assert rep == json.loads(json.dumps(res.report))
    assert rep["folder"] == TASK and rep["rig_id"] == RIG and rep["slam_backend"] == "oracle"
    assert rep["range"] == ["episode_000001", "episode_000004"] and rep["exit_code"] == 0
    assert rep["totals"] == {"episode": 3, "failed": 1, "waiting": 0, "skipped": 0}
    assert rep["calibration"]["state"] == "done" and rep["calibration"]["ran"] is False      # never fitted again
    assert [r["source_name"] for r in rep["recordings"]] == ["episode_000001", BAD, "episode_000003",
                                                             "episode_000004"]


def test_renamed_episodes_stay_up_to_date(done, tmp_path):
    """The stage hashes do not depend on the dir path: after .staging/x -> episode_N nothing needs re-running."""
    root = copy_root(done[0], tmp_path / "root")
    pr = run_process(root, TASK, slam="oracle", log=quiet)
    assert pr.ran == {"episode_1": [], "episode_2": [], "episode_3": []}


def test_rerun_is_a_noop(work, capsys):
    root, src = work
    td = root.tasks / TASK
    before = {k: v for k, v in tree_digest(root.path).items() if k != f"tasks/{TASK}/report.json"}
    res = _run(root, src, start="episode_000000")
    assert res.exit_code == EXIT_OK and all(o.state == "skipped" for o in res.outcomes)
    oc = _outcomes(res)
    assert all(oc[n].quiet for n in GOOD) and oc[BAD].message.startswith("failed earlier at slam: oracle_no_truth")
    assert {k: v for k, v in tree_digest(root.path).items() if k != f"tasks/{TASK}/report.json"} == before
    assert cli_run(root, TASK, "episode_000000", "episode_000004", source=src, slam="oracle") == 0
    out = capsys.readouterr().out
    assert "(3 already episodes of this folder" in out and "failed earlier at slam: oracle_no_truth" in out
    assert "dataproc package" in out and (td / "report.json").is_file()


def test_dry_run_changes_nothing(work):
    root, src = work
    before = tree_digest(root.path)
    res = _run(root, src, retry_failed=True, dry_run=True)
    plan = {r.name: r for r in res.plan.recordings}
    assert plan[BAD].action == RETRY and plan[BAD].why == "failed earlier at slam: oracle_no_truth"
    assert all(plan[n].action == SKIP and plan[n].quiet for n in GOOD)
    assert tree_digest(root.path) == before


def test_new_recordings_continue_the_numbering(work, tmp_path_factory):
    """A recording that finished syncing later is picked up by the same range; numbers continue after the last."""
    root, _ = work
    src = tmp_path_factory.mktemp("src_more")
    standard_source(tmp_path_factory, src, bad=(2,))
    pi_source(pristine(tmp_path_factory)["demos"][0], src / "episode_000005", 5)
    res = _run(root, src, end="episode_000005")
    assert _outcomes(res)["episode_000005"].slot == "episode_4"


def test_package_after_run_has_exactly_the_episodes(work):
    from dataproc.package import package_dataset

    root, _ = work
    res = package_dataset(root, "practice", [TASK], log=quiet)
    sel = res.selection
    assert [e.slot for e in sel.episodes] == ["episode_1", "episode_2", "episode_3"]
    assert [x.entry for x in sel.excluded] == [f"failed/{p.name}" for p in (root.tasks / TASK / "failed").iterdir()]
    from dataproc._vendor.replay_buffer import ReplayBuffer

    rb = ReplayBuffer.create_from_path(str(res.zip_path))
    assert rb.n_episodes == 3 and list(rb.meta["task_names"][:]) == [TASK] * 3


# --------------------------------------------------------------------------------------------- retry and crashes

def _one_recording_root(tmp_path_factory, tmp_path):
    """A calibrated copy and a source with the calibration and one good demo (``episode_000001``)."""
    root = copy_root(calibrated_root(tmp_path_factory)[0], tmp_path / "root")
    src = tmp_path / "src"
    p = pristine(tmp_path_factory)
    pi_source(p["cal"], src / "episode_000000", 0)
    pi_source(p["demos"][0], src / "episode_000001", 1)
    return root, src


def test_retry_failed(tmp_path_factory, tmp_path, monkeypatch):
    from dataproc.stages.geometry import PoseStage

    root, src = _one_recording_root(tmp_path_factory, tmp_path)
    td = root.tasks / TASK

    def boom(self, ctx, ep):
        raise ZeroDivisionError("synthetic bug")

    with monkeypatch.context() as m:
        m.setattr(PoseStage, "run", boom)
        res = _run(root, src, end="episode_000001")
    o = _outcomes(res)["episode_000001"]
    assert o.state == "failed" and o.stage == "pose" and o.reasons == ["exception"]
    assert (td / "failed" / o.pi_name / "processed" / "error.txt").is_file()
    # without --retry-failed: skipped, with the reason
    res = _run(root, src, end="episode_000001")
    assert _outcomes(res)["episode_000001"].message.startswith("failed earlier at pose: exception")
    # --retry-failed re-runs only what needs it: nothing changed, so it fails again without running a stage
    res = _run(root, src, end="episode_000001", retry_failed=True)
    o = _outcomes(res)["episode_000001"]
    assert o.state == "failed" and o.ran == [] and (td / "failed" / o.pi_name / "reason.json").is_file()
    # --retry-failed --force: every stage again, the bug is gone -> episode_1
    res = _run(root, src, end="episode_000001", retry_failed=True, force=True)
    o = _outcomes(res)["episode_000001"]
    assert o.state == "episode" and o.slot == "episode_1" and o.ran == list(STAGE_NAMES)
    assert not (td / "failed" / o.pi_name).exists() and not (td / "episode_1" / "reason.json").exists()
    assert {r["source_name"]: r["status"] for r in read_sources(root.path / SOURCES_FILE)}["episode_000001"] == \
        "ingested"


def test_crash_mid_episode_resumes_from_staging(tmp_path_factory, tmp_path, monkeypatch):
    from dataproc.stages.output import ImagesStage

    root, src = _one_recording_root(tmp_path_factory, tmp_path)
    td = root.tasks / TASK

    def crash(self, ctx, ep):
        raise KeyboardInterrupt                          # the process dies: nothing catches it

    with monkeypatch.context() as m:
        m.setattr(ImagesStage, "run", crash)
        with pytest.raises(KeyboardInterrupt):
            _run(root, src, end="episode_000001")
    (staged,) = list((td / ".staging").iterdir())
    st = load_status(staged / "processed")
    assert st["validate"]["status"] == "done" and "images" not in st
    plan = plan_run(root, TASK, "episode_000000", "episode_000001", src)
    assert {r.name: r.action for r in plan.recordings}["episode_000001"] == RESUME
    res = _run(root, src, end="episode_000001")
    o = _outcomes(res)["episode_000001"]
    assert o.state == "episode" and o.slot == "episode_1" and o.ran == ["images", "export"]


def test_crash_after_promotion_rebuilds_the_ledger_row(tmp_path_factory, tmp_path, monkeypatch):
    from dataproc.intake import Ledger

    root, src = _one_recording_root(tmp_path_factory, tmp_path)
    td = root.tasks / TASK
    orig = Ledger.append

    def die_on_episode(self, row):
        if row["slot"].startswith("episode_"):
            raise KeyboardInterrupt
        return orig(self, row)

    with monkeypatch.context() as m:
        m.setattr(Ledger, "append", die_on_episode)
        with pytest.raises(KeyboardInterrupt):
            _run(root, src, end="episode_000001")
    assert (td / "episode_1" / "raw").is_dir()
    assert "episode_1" not in {r["slot"] for r in read_ledger(td / "episodes.csv")}
    res = _run(root, src, end="episode_000001")
    o = _outcomes(res)["episode_000001"]
    assert o.state == "skipped" and o.quiet and o.message == "already episode_1 of this folder"
    rows = [r for r in read_ledger(td / "episodes.csv") if r["slot"] == "episode_1"]
    assert len(rows) == 1 and rows[0]["pi_name"] == o.pi_name


def test_crash_before_reason_json_is_repaired(tmp_path_factory, tmp_path, monkeypatch):
    import dataproc.run.staging as S

    root0, _ = calibrated_root(tmp_path_factory)
    root = copy_root(root0, tmp_path / "root")
    src = tmp_path / "src"
    standard_source(tmp_path_factory, src, bad=(2,))
    orig = S.write_json

    def die_on_reason(path, obj):
        if path.name == S.REASON_FILE:
            raise KeyboardInterrupt
        return orig(path, obj)

    with monkeypatch.context() as m:
        m.setattr(S, "write_json", die_on_reason)
        with pytest.raises(KeyboardInterrupt):
            _run(root, src, start=BAD, end=BAD)
    (failed,) = list((root.tasks / TASK / "failed").iterdir())
    assert not (failed / "reason.json").exists()
    res = _run(root, src, start=BAD, end=BAD)
    rec = json.loads((failed / "reason.json").read_text())
    assert rec["stage"] == "slam" and rec["reasons"] == ["oracle_no_truth"] and "rebuilt" in rec["note"]
    assert _outcomes(res)[BAD].message.startswith("failed earlier at slam: oracle_no_truth")
    assert {r["source_name"]: r["status"] for r in read_sources(root.path / SOURCES_FILE)}[BAD] == "failed"


def test_force_with_no_slam_backend_is_refused(tmp_path_factory):
    root, src = calibrated_root(tmp_path_factory)
    with pytest.raises(RunError, match="--force re-runs slam"):
        _run(root, src, slam="none", force=True)
