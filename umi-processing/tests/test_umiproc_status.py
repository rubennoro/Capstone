"""``umiproc status``: the data-root summary and one task folder, in plain words, read only."""

import contextlib
import io
import json

import pytest
import yaml

from umiproc import recordings as R
from umiproc.cli import main
from umiproc.dp.files import save_status, write_json
from umiproc.dp.paths import DataRoot
from umiproc.dp.stages.calibration import GRIPPER_RANGE_FILE
from umiproc.lens.lensfile import dump_lens, lens_document
from umiproc.paths import rig_file, rig_lens_file
from umiproc.sensors import ensure_rig_files
from umiproc_testlib import pi_recording, tree_digest

RIG = "rig_pi_v3"
quiet = lambda *a, **k: None  # noqa: E731


def _status(*argv, env_root=None):
    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        if env_root is not None:
            mp.setenv("UMIPROC_ROOT", str(env_root))
        else:
            mp.delenv("UMIPROC_ROOT", raising=False)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["status", *argv])
    return code, out.getvalue(), err.getvalue()


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """A root with copies, a lens file, a session folder (map, calibration recording, one demo waiting, one failed,
    one episode) and a second map's folder."""
    base = tmp_path_factory.mktemp("status")
    src, root = base / "src", DataRoot(base / "root")
    src.mkdir()
    for n in range(1, 7):
        pi_recording(src, n)
    ensure_rig_files(root.path, RIG, log=quiet)
    m = R.take_recording(root, src, "episode_000001", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    R.take_recording(root, src, "episode_000002", R.CALIBRATION, rig_id=RIG, folder=m.folder, log=quiet)
    d = [R.take_recording(root, src, f"episode_00000{n}", R.DEMONSTRATION, rig_id=RIG, folder=m.folder, log=quiet)
         for n in (3, 4, 5)]
    f = m.task_dir
    # demo 4 failed at slam; demo 5 became episode_1 (as the run step will move them)
    (f / "failed").mkdir()
    (f / ".staging" / d[1].pi_name).rename(f / "failed" / d[1].pi_name)
    write_json(f / "failed" / d[1].pi_name / "reason.json",
               {"stage": "slam", "reasons": ["too_many_lost_frames"], "message": "12 lost frames"})
    (f / ".staging" / d[2].pi_name).rename(f / "episode_1")
    (f / "episode_1" / "processed").mkdir()
    save_status(f / "episode_1" / "processed", {s: {"status": "done"} for s in
                                                  ("check", "slam", "grid", "pose", "width", "validate", "images")}
                | {"export": {"status": "done", "outputs": {"n_steps": 287}}})
    R.take_recording(root, src, "episode_000006", R.MAP, rig_id=RIG, task="block_raise", log=quiet)
    return root, m, d


def test_root_summary(session):
    root, m, d = session
    before = tree_digest(root.path)
    code, out, err = _status(env_root=root.path)
    assert code == 0, err
    assert tree_digest(root.path) == before                               # read only
    assert f"data root: {root.path} (from $UMIPROC_ROOT)" in out
    assert "rig_pi_v3.yaml: same as the repo's" in out and "rig_pi_v3.slam.yaml: same as the repo's" in out
    assert 'lens file:     none, so SLAM uses the rig file\'s nominal lens model ("nominal, uncalibrated")' in out
    assert "umiproc_slam:" in out and "vocabulary:" in out
    assert "recordings taken (pi_sources.csv): 2 map, 1 calibration, 3 demonstration" in out
    assert ("block_raise: label block_raise, map yes, gripper calibration no, 1 episode, 1 waiting, 1 failed"
            in out)
    assert "block_raise_2: label block_raise, map yes, gripper calibration no, 0 episodes" in out


def test_folder_details(session):
    root, m, d = session
    before = tree_digest(root.path)
    code, out, err = _status("block_raise", "--root", str(root.path))
    assert code == 0, err
    assert tree_digest(root.path) == before
    assert f"task folder {m.task_dir} (label block_raise, rig rig_pi_v3" in out
    assert f"map:                  yes (map/{m.pi_name})" in out
    assert "gripper calibration:  no" in out
    assert "episodes:             1" in out and "episode_1" in out and "exported" in out and "287" in out
    assert f"{d[0].pi_name}: waiting for the gripper calibration" in out
    assert f"{d[1].pi_name}: slam: too_many_lost_frames" in out
    code2, out2, _ = _status(str(m.task_dir), env_root=root.path)         # a path works too
    assert code2 == 0 and out2 == out


def test_gripper_calibration_and_lens_shown(tmp_path, session):
    root, m, d = session
    import shutil

    copy = tmp_path / "root"
    shutil.copytree(root.path, copy)
    f = copy / "tasks" / "block_raise"
    (f / GRIPPER_RANGE_FILE).write_text(json.dumps({"min_width": 0.04, "max_width": 0.12}))
    rig_lens_file(copy, RIG).write_text(dump_lens(lens_document(
        rig_id=RIG, sensor="imx219", resolution=(1280, 960), intrinsics=[580.0, 580.0, 640.0, 480.0],
        distortion=[0.02, -0.005, 0.001, 0.0], date="2026-11-16T18:00:00Z",
        source={"pi_name": "pi-2026-11-16_episode_000123"},
        board={}, metrics={})))
    d0 = yaml.safe_load(rig_file(copy, RIG).read_text())
    d0["imu"]["noise"]["gyroscope_random_walk"] = 3e-6
    rig_file(copy, RIG).write_text(yaml.safe_dump(d0, sort_keys=False))
    code, out, _ = _status("--root", str(copy))
    assert ("lens file:     rig_pi_v3.lens.yaml: calibrated 2026-11-16T18:00:00Z from "
            "pi-2026-11-16_episode_000123") in out
    assert "differs from the repo's" in out and "imu.noise.gyroscope_random_walk" in out
    assert "block_raise: label block_raise, map yes, gripper calibration yes" in out
    code, out, _ = _status("block_raise", "--root", str(copy))
    assert "gripper calibration:  yes, gripper_range.json" in out
    assert f"{d[0].pi_name}: not finished" in out


def test_empty_and_missing_roots(tmp_path):
    code, out, _ = _status("--root", str(tmp_path / "nothing"))
    assert code == 0 and "does not exist yet" in out and "not there yet" in out
    assert not (tmp_path / "nothing").exists()
    (tmp_path / "empty").mkdir()
    code, out, _ = _status("--root", str(tmp_path / "empty"))
    assert code == 0 and "recordings taken (pi_sources.csv): none yet" in out and "none yet (`umiproc map`" in out
    assert list((tmp_path / "empty").iterdir()) == []


def test_errors(tmp_path):
    code, _, err = _status("--root", "/data")
    assert code == 1 and "refused" in err
    code, out, _ = _status("no_such_folder", "--root", str(tmp_path))
    assert code == 1 and "no task folder" in out
    code, _, err = _status("bad name!", "--root", str(tmp_path))
    assert code == 1 and "umiproc status:" in err


def test_problem_with_the_rig_files_is_shown(tmp_path):
    ensure_rig_files(tmp_path, RIG, log=quiet)
    d0 = yaml.safe_load(rig_file(tmp_path, RIG).read_text())
    d0["imu"]["time_offset_ns"] = 5
    rig_file(tmp_path, RIG).write_text(yaml.safe_dump(d0, sort_keys=False))
    code, out, _ = _status("--root", str(tmp_path))
    assert code == 0 and "PROBLEM:" in out and "time offset of 0" in out


def test_a_rig_copy_the_processing_refuses_is_shown(tmp_path):
    ensure_rig_files(tmp_path, RIG, log=quiet)
    d0 = yaml.safe_load(rig_file(tmp_path, RIG).read_text())
    del d0["gripper"]
    rig_file(tmp_path, RIG).write_text(yaml.safe_dump(d0, sort_keys=False))
    code, out, _ = _status("--root", str(tmp_path))
    assert code == 0 and "PROBLEM:" in out and "no gripper section" in out
