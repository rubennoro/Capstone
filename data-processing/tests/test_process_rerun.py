"""Re-runs: skipping by inputs_hash, --force, a rig change re-running only the affected stages, --episode N,
and the fatal (exit 1) setup errors."""

import json
import shutil

import numpy as np
import pytest

from dataproc.files import load_status
from dataproc.process import run_process
from dataproc.report import STAGE_NAMES as STAGES
from dataproc.stages.geometry import SlamStage
from dataproc._vendor.replay_buffer import ReplayBuffer
from test_process_common import (N_DEMOS, TASK, copy_root, fast_rig_dict, processed_root, quiet, task_dir,
                                 template_root, write_rig)


@pytest.fixture
def root(tmp_path_factory, tmp_path):
    return copy_root(processed_root(tmp_path_factory)[0], tmp_path / "root")


def _starts(root):
    return {k: {s: r["t_start"] for s, r in load_status(task_dir(root) / f"episode_{k}" / "processed").items()}
            for k in range(N_DEMOS)}


def _calib_start(root):
    p = task_dir(root) / "calibration" / "cal_episode_000000" / "processed" / "status.json"
    return json.loads(p.read_text())["gripper_range"]["t_start"]


def test_second_run_skips_everything(root):
    before, cal = _starts(root), _calib_start(root)
    res = run_process(root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0 and res.zip_path.is_file()
    assert res.ran == {f"episode_{k}": [] for k in range(N_DEMOS)}
    assert _starts(root) == before and _calib_start(root) == cal
    assert res.report["calibration"]["ran"] is False
    assert res.report["duration_s"] < 15                   # only the zip rebuild


def test_force_reruns_everything(root):
    before, cal = _starts(root), _calib_start(root)
    res = run_process(root, TASK, slam="oracle", force=True, log=quiet)
    assert res.exit_code == 0
    assert res.ran == {f"episode_{k}": list(STAGES) for k in range(N_DEMOS)}
    after = _starts(root)
    assert all(after[k][s] > before[k][s] for k in after for s in STAGES)
    assert _calib_start(root) > cal and res.report["calibration"]["ran"] is True


def test_world_offset_change_reruns_pose_and_later(root):
    td = task_dir(root)
    z = td / "output" / f"{TASK}.zarr.zip"
    pos0 = ReplayBuffer.create_from_path(str(z)).data["gripper_right_eef_pos"][:]
    before = _starts(root)
    d = fast_rig_dict()
    d["world_offset"] = {"translation_m": [0.5, -0.2, 0.1], "yaw_deg": 75.0}
    write_rig(root, d)
    res = run_process(root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0
    later = ["pose", "validate", "images", "export"]
    assert res.ran == {f"episode_{k}": later for k in range(N_DEMOS)}
    after = _starts(root)
    for k in range(N_DEMOS):
        for s in STAGES:
            assert (after[k][s] > before[k][s]) == (s in later), (k, s)
    pos1 = ReplayBuffer.create_from_path(str(z)).data["gripper_right_eef_pos"][:]
    assert pos1.shape == pos0.shape and np.abs(pos1 - pos0).max() > 0.1
    # relative motion is unchanged (the offset is a left-multiplied rigid transform)
    assert np.allclose(np.linalg.norm(np.diff(pos1, axis=0), axis=1), np.linalg.norm(np.diff(pos0, axis=0), axis=1),
                       atol=1e-5)


def test_label_change_reruns_images_and_export_only(root):
    import yaml

    ty = task_dir(root) / "task.yaml"
    d = yaml.safe_load(ty.read_text())
    d["label"] = "eraser_lift"
    ty.write_text(yaml.safe_dump(d))
    res = run_process(root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0
    assert res.ran == {f"episode_{k}": ["images", "export"] for k in range(N_DEMOS)}
    rb = ReplayBuffer.create_from_path(str(res.zip_path))
    assert rb.meta["task_names"][:].tolist() == ["eraser_lift"] * N_DEMOS


def test_single_episode_then_zip_rebuilt(root):
    before = _starts(root)
    res = run_process(root, TASK, episode=1, force=True, slam="oracle", log=quiet)
    assert res.exit_code == 0 and res.ran == {"episode_1": list(STAGES)}
    after = _starts(root)
    assert after[0] == before[0] and after[2] == before[2] and after[1]["check"] > before[1]["check"]
    assert res.report["zip"]["n_episodes"] == N_DEMOS and len(res.report["episodes"]) == N_DEMOS
    assert ReplayBuffer.create_from_path(str(res.zip_path)).n_episodes == N_DEMOS


def test_single_episode_missing(root):
    res = run_process(root, TASK, episode=7, slam="oracle", log=quiet)
    assert res.exit_code == 1 and "episode_7" in res.error and res.report is None


def test_fatal_setup_errors(tmp_path_factory, tmp_path, capsys):
    from dataproc import cli
    import os

    root = copy_root(template_root(tmp_path_factory), tmp_path / "root")
    os.environ["DATAPROC_ROOT"] = str(root.path)
    try:
        assert cli.main(["process", "--task", "other", "--slam", "oracle"]) == 1
        assert "run intake first" in capsys.readouterr().err
        assert not (root.tasks / "other").exists()
        (root.rigs / "rig_v1_fast.yaml").rename(root.rigs / "moved.yaml")
        assert cli.main(["process", "--task", TASK, "--slam", "oracle"]) == 1
        assert "rig_v1_fast.yaml is missing" in capsys.readouterr().err
        assert not (task_dir(root) / "output").exists() and not (task_dir(root) / "episode_0" / "processed").exists()
        (root.rigs / "moved.yaml").rename(root.rigs / "rig_v1_fast.yaml")
        (task_dir(root) / "task.yaml").unlink()
        assert cli.main(["process", "--task", TASK, "--slam", "oracle"]) == 1
        assert "task.yaml is missing" in capsys.readouterr().err
    finally:
        del os.environ["DATAPROC_ROOT"]


def test_status_json_records(root):
    st = load_status(task_dir(root) / "episode_0" / "processed")
    for s in STAGES:
        r = st[s]
        assert r["status"] == "done" and len(r["inputs_hash"]) == 64 and r["t_end"] >= r["t_start"]
        assert r["version"] == (SlamStage.version if s == "slam" else 1) and r["reason"] is None and r["reasons"] == [] and isinstance(r["outputs"], dict)
    assert st["export"]["outputs"]["n_steps"] == 150
    assert len({st[s]["inputs_hash"] for s in STAGES}) == len(STAGES)
    shutil.rmtree(task_dir(root) / "episode_0" / "processed")          # lost processed/: everything reruns
    res = run_process(root, TASK, slam="oracle", log=quiet)
    assert res.ran["episode_0"] == list(STAGES) and res.ran["episode_1"] == []
