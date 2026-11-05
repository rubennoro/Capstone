"""Fault cases: every bad episode is recorded with its reason and never stops the run."""

import json
import shutil

import pytest

from dataproc.files import load_status
from dataproc.process import EpisodeProcessor, run_process
from dataproc.report import STAGE_NAMES
from dataproc.stages.geometry import PoseStage
from dataproc.slam.oracle import OracleBackend
from dataproc._vendor.replay_buffer import ReplayBuffer
from test_process_common import (N_DEMOS, TASK, add_demo, copy_root, fast_rig_dict, processed_root, quiet, task_dir,
                                 template_root, write_rig)


@pytest.fixture
def done_root(tmp_path_factory, tmp_path):
    return copy_root(processed_root(tmp_path_factory)[0], tmp_path / "root")


@pytest.fixture
def fresh_root(tmp_path_factory, tmp_path):
    return copy_root(template_root(tmp_path_factory), tmp_path / "root")


def _status(root, k):
    return load_status(task_dir(root) / f"episode_{k}" / "processed")


def _ep(res, slot):
    return next(e for e in res.report["episodes"] if e["slot"] == slot)


def test_imu_gap_rejected(done_root):
    add_demo(done_root, "gap_episode_000000", seed=77, faults={"imu_gaps": [(3.0, 0.05)]})   # 50 ms, mid-span
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0 and res.ran["episode_3"] and res.ran["episode_0"] == []
    st = _status(done_root, 3)
    assert st["check"]["status"] == "done"                     # a gap is a validator warning, not an error
    assert st["validate"]["status"] == "rejected" and "imu_gap" in st["validate"]["reasons"]
    v = json.loads((task_dir(done_root) / "episode_3" / "processed" / "validate.json").read_text())
    assert v["metrics"]["max_imu_gap_ms"] > 20
    assert st["images"]["status"] == "skipped" and st["export"]["status"] == "skipped"
    e = _ep(res, "episode_3")
    assert e["status"] == "rejected" and "validate: imu_gap" in e["reasons"]
    assert res.report["zip"]["n_episodes"] == N_DEMOS and res.report["totals"]["rejected"] == 1
    # rejected + unchanged inputs: not retried
    res2 = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res2.ran["episode_3"] == []


def test_noisy_slam_scale_rejected(done_root):
    z = task_dir(done_root) / "output" / f"{TASK}.zarr.zip"
    assert z.is_file()
    res = run_process(done_root, TASK, slam="oracle", force=True, backend=OracleBackend(noise_pos_m=0.03),
                      log=quiet)
    for k in range(N_DEMOS):
        st = _status(done_root, k)
        assert st["validate"]["status"] == "rejected"
        assert any(r.startswith("scale_") for r in st["validate"]["reasons"]), st["validate"]["reasons"]
    assert res.exit_code == 1 and res.zip_path is None and z.is_file()        # old zip kept (stale)
    assert res.report["zip"] is None
    assert res.report["zip_note"] == "no exportable episodes; previous zip kept and is stale"


def test_missing_calibration(fresh_root):
    td = task_dir(fresh_root)
    cal = td / "calibration"
    saved = fresh_root.path / "saved_calibration"
    shutil.move(str(cal), str(saved))
    res = run_process(fresh_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 1 and res.zip_path is None
    assert not (td / "output" / f"{TASK}.zarr.zip").exists() and (td / "output" / "report.json").is_file()
    assert res.report["calibration"]["state"] == "no_calibration"
    for k in range(N_DEMOS):
        st = _status(fresh_root, k)
        assert [st[s]["status"] for s in STAGE_NAMES] == ["done"] * 4 + ["failed"] + ["skipped"] * 3
        assert st["width"]["reasons"] == ["no_calibration"]
        assert _ep(res, f"episode_{k}")["status"] == "failed"
    # calibration arrives: only width and later run
    shutil.move(str(saved), str(cal))
    res = run_process(fresh_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0
    assert res.ran == {f"episode_{k}": ["width", "validate", "images", "export"] for k in range(N_DEMOS)}


def test_failed_calibration(fresh_root):
    """Calibration tags never found (the rig's aruco ids changed) -> calibration_failed for every width."""
    d = fast_rig_dict()
    d["aruco"]["left_id"], d["aruco"]["right_id"] = 5, 6
    write_rig(fresh_root, d)
    res = run_process(fresh_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 1 and res.report["calibration"]["state"] == "calibration_failed"
    cal = task_dir(fresh_root) / "calibration" / "cal_episode_000000" / "processed"
    assert load_status(cal)["gripper_range"]["reasons"] == ["gripper_range_fit"]
    assert not (cal / "gripper_range.json").exists()
    assert not (task_dir(fresh_root) / "gripper_range.json").exists()
    for k in range(N_DEMOS):
        assert _status(fresh_root, k)["width"]["reasons"] == ["calibration_failed"]


def test_rig_resolution_mismatch_rejected_at_check(fresh_root):
    d = fast_rig_dict()
    d["cameras"]["main"]["resolution"] = [322, 182]
    write_rig(fresh_root, d)
    res = run_process(fresh_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 1
    for k in range(N_DEMOS):
        st = _status(fresh_root, k)
        assert st["check"]["status"] == "rejected" and st["check"]["reasons"] == ["rig_resolution_mismatch"]
        assert all(st[s]["status"] == "skipped" for s in STAGE_NAMES[1:])
    chk = json.loads((task_dir(fresh_root) / "episode_0" / "processed" / "check.json").read_text())
    assert chk["rig_resolution_mismatch"][0]["meta"] == [320, 180]


def test_unexpected_exception_isolated(done_root, monkeypatch):
    orig = PoseStage.run

    def boom(self, ctx, ep):
        if ep.slot == "episode_1":
            raise ZeroDivisionError("synthetic bug")
        return orig(self, ctx, ep)

    monkeypatch.setattr(PoseStage, "run", boom)
    d = fast_rig_dict()
    d["world_offset"] = {"translation_m": [0.0, 0.0, 0.0], "yaw_deg": 45.0}   # pose and later rerun
    write_rig(done_root, d)
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0
    st = _status(done_root, 1)
    assert st["pose"]["status"] == "failed" and st["pose"]["reasons"] == ["exception"]
    assert "ZeroDivisionError: synthetic bug" in st["pose"]["reason"]
    err = (task_dir(done_root) / "episode_1" / "processed" / "error.txt").read_text()
    assert "stage pose" in err and "Traceback" in err and "ZeroDivisionError" in err
    assert ReplayBuffer.create_from_path(str(res.zip_path)).meta["episode_names"][:].tolist() == [
        "demo_episode_000000", "demo_episode_000002"]
    monkeypatch.setattr(PoseStage, "run", orig)
    assert run_process(done_root, TASK, slam="oracle", log=quiet).ran["episode_1"] == []    # same hash: no retry
    res = run_process(done_root, TASK, episode=1, force=True, slam="oracle", log=quiet)
    assert _ep(res, "episode_1")["status"] == "exported" and res.report["zip"]["n_episodes"] == N_DEMOS
    assert not (task_dir(done_root) / "episode_1" / "processed" / "error.txt").exists()


def test_oracle_without_truth(done_root):
    raw = task_dir(done_root) / "episode_2" / "raw"
    (raw / "truth.npz").unlink()                 # a "real" episode: no truth, manifest consistent
    m = raw / "MANIFEST.sha256"
    m.write_text("".join(ln + "\n" for ln in m.read_text().splitlines() if not ln.endswith("truth.npz")))
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    st = _status(done_root, 2)
    assert st["check"]["status"] == "done"
    assert st["slam"]["status"] == "failed" and st["slam"]["reasons"] == ["oracle_no_truth"]
    assert "truth.npz" in st["slam"]["reason"] and "orbslam3" in st["slam"]["reason"]
    assert res.exit_code == 0 and res.report["zip"]["n_episodes"] == 2


def test_corrupt_raw_rejected(done_root):
    raw = task_dir(done_root) / "episode_0" / "raw"
    f = raw / "imu.csv"
    f.write_bytes(f.read_bytes() + b"\n")          # bytes changed after intake, MANIFEST unchanged
    # inputs_hash covers MANIFEST.sha256, not the data files: an unchanged manifest means no recheck ...
    assert run_process(done_root, TASK, slam="oracle", log=quiet).ran["episode_0"] == []
    # ... --force re-verifies the manifest and rejects
    res = run_process(done_root, TASK, slam="oracle", force=True, log=quiet)
    st = _status(done_root, 0)
    assert st["check"]["status"] == "rejected" and "hash_mismatch" in st["check"]["reasons"]
    assert res.exit_code == 0 and res.report["zip"]["n_episodes"] == 2


def test_stage_order_matches_report_names():
    assert tuple(cls.name for cls in EpisodeProcessor.STAGES) == STAGE_NAMES


def test_build_failure_keeps_previous_zip(done_root, monkeypatch, capsys):
    """Reviewer S5: the zip build fails (ENOSPC) -> the old zip is untouched, exit 1, status says stale."""
    import dataproc.export.zarr_writer as ZW
    from dataproc import cli

    td = task_dir(done_root)
    z = td / "output" / f"{TASK}.zarr.zip"
    before = z.read_bytes()

    def enospc(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ZW, "build_task_zarr_zip", enospc)
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 1 and res.zip_path is None and z.read_bytes() == before
    assert res.report["zip"] is None
    assert "No space left on device" in res.report["zip_note"] and "previous zip kept and is stale" in res.report[
        "zip_note"]
    assert "OSError" in (td / "output" / "error.txt").read_text()
    monkeypatch.setenv("DATAPROC_ROOT", str(done_root.path))
    assert cli.main(["status", "--task", TASK]) == 0
    assert "stale" in capsys.readouterr().out
    monkeypatch.undo()
    monkeypatch.setenv("DATAPROC_ROOT", str(done_root.path))
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0 and not (td / "output" / "error.txt").exists()
    assert cli.main(["status", "--task", TASK]) == 0
    assert "up to date" in capsys.readouterr().out


def test_nothing_exportable_keeps_previous_zip(done_root, monkeypatch, capsys):
    """Reviewer S6: calibration removed after a good run -> width fails everywhere; the old zip is kept, stale."""
    from dataproc import cli

    td = task_dir(done_root)
    z = td / "output" / f"{TASK}.zarr.zip"
    before = z.read_bytes()
    shutil.move(str(td / "calibration"), str(done_root.path / "cal_aside"))
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 1 and z.read_bytes() == before
    assert res.report["zip"] is None
    assert res.report["zip_note"] == "no exportable episodes; previous zip kept and is stale"
    assert all(e["status"] == "failed" for e in res.report["episodes"])
    monkeypatch.setenv("DATAPROC_ROOT", str(done_root.path))
    assert cli.main(["status", "--task", TASK]) == 0
    out = capsys.readouterr().out
    assert "stale" in out and "0 exported" in out


def test_status_stale_by_episode_names(done_root, monkeypatch, capsys):
    """The report says the zip was built, but the episodes exported now differ from the zip's episode_names."""
    from dataproc import cli
    from dataproc.files import save_status

    p = task_dir(done_root) / "episode_2" / "processed"
    st = load_status(p)
    st["export"]["status"] = "failed"
    save_status(p, st)
    monkeypatch.setenv("DATAPROC_ROOT", str(done_root.path))
    assert cli.main(["status", "--task", TASK]) == 0
    assert "stale" in capsys.readouterr().out


def test_calibration_unreadable_saved_range_refits(done_root):
    """A saved gripper_range.json that fails to load in any way (here a KeyError) is fitted again; a leftover
    calibration error.txt is removed by the successful fit; the episodes are unaffected (same range bytes)."""
    cal = task_dir(done_root) / "calibration" / "cal_episode_000000" / "processed"
    good = (cal / "gripper_range.json").read_text()
    (cal / "gripper_range.json").write_text(json.dumps({
        "left_finger_tag_id": 0, "right_finger_tag_id": 1, "max_width": 1.0, "min_width": 0.0,
        "left_to_right": {}, "right_to_left": {}}))
    (cal / "error.txt").write_text("old traceback")
    res = run_process(done_root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0 and res.report["calibration"]["ran"] is True
    assert (cal / "gripper_range.json").read_text() == good and not (cal / "error.txt").exists()
    assert res.ran == {f"episode_{k}": [] for k in range(N_DEMOS)}
