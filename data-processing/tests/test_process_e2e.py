"""End to end: sim (1 calibration + 3 demos, --fast) -> intake -> process (oracle) -> the exported episodes (in a
zip built as ``dataproc package`` does), checked against truth."""

import json
import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dataproc import cli
from dataproc.geometry.trajectory import Trajectory
from dataproc.intake import read_ledger
from dataproc.files import load_status
from dataproc.process import run_process
from dataproc.report import STAGE_NAMES as STAGES
from dataproc.sim.episode import read_truth
from dataproc.stages.width import GripperRange
from test_process_common import (N_DEMOS, TASK, T_from, copy_root, processed_root, quiet, task_dir, task_zip,
                                 template_root, tree_digest)


@pytest.fixture(scope="module")
def done(tmp_path_factory):
    return processed_root(tmp_path_factory)


def test_report(done):
    root, res = done
    td = task_dir(root)
    assert res.exit_code == 0 and not (td / "output" / f"{TASK}.zarr.zip").exists()     # package builds zips
    rep = json.loads((td / "output" / "report.json").read_text())
    assert rep == res.report
    assert rep["totals"]["exported"] == N_DEMOS and rep["totals"]["episodes"] == N_DEMOS
    assert rep["calibration"]["state"] == "done" and rep["calibration"]["name"] == "cal_episode_000000"
    assert rep["rig_id"] == "rig_v1_fast" and "tcp" in rep["rig_nominal_sections"]
    assert rep["slam_backend"] == "oracle" and rep["dataproc_version"] and rep["created"]
    assert "zip" not in rep
    for e in rep["episodes"]:
        assert e["status"] == "exported" and e["reasons"] == [] and e["n_steps"] == 150
        assert set(e["stages"].values()) == {"done"}
    assert (td / "gripper_range.json").read_bytes() == (
        td / "calibration" / "cal_episode_000000" / "processed" / "gripper_range.json").read_bytes()


def test_replay_buffer_contents(done):
    root, _ = done
    td = task_dir(root)
    rb = task_zip(root)
    ledger = [r for r in read_ledger(td / "episodes.csv") if r["kind"] == "demonstration"]
    assert rb.n_episodes == N_DEMOS
    assert rb.meta["episode_names"][:].tolist() == [r["pi_name"] for r in ledger]
    assert rb.meta["task_names"][:].tolist() == [TASK] * N_DEMOS
    assert rb.episode_ends[:].tolist() == [150, 300, 450]
    for k in ("camera_right_main_rgb", "camera_right_ultrawide_rgb", "gripper_right_eef_pos",
              "gripper_right_eef_rot_axis_angle", "gripper_right_gripper_width"):
        assert k in rb.data
    assert rb.data["camera_right_main_rgb"].shape == (450, 224, 224, 3)


def _truth_tcp(ep_raw, grid, rig_T_imu_cam1, T_cam_tcp):
    tr = read_truth(ep_raw)
    T, ok = Trajectory.from_T(tr["t_ns"], tr["T_W_imu"]).sample(grid, max_gap_ns=5_000_000)
    assert ok.all()
    w = np.interp(grid.astype(np.float64), tr["t_ns"].astype(np.float64), tr["width_m"])
    return T @ rig_T_imu_cam1 @ T_cam_tcp, w


def test_pose_and_width_vs_truth(done):
    """Relative TCP poses (world invariant) within 1 mm / 0.1 deg; width = truth - min_width within 2 mm."""
    from dataproc.config import load_rig

    root, _ = done
    td = task_dir(root)
    rig = load_rig(root.rigs / "rig_v1_fast.yaml")
    gr = GripperRange.load(td / "gripper_range.json")
    rb = task_zip(root)
    for k in range(N_DEMOS):
        ep = td / f"episode_{k}"
        v = json.loads((ep / "processed" / "validate.json").read_text())["kept"]
        grid = np.load(ep / "processed" / "grid.npy")[v["i0"]:v["i1"]]
        ends = [0] + rb.episode_ends[:].tolist()
        e = {key: rb.data[key][ends[k]:ends[k + 1]] for key in ("gripper_right_eef_pos",
             "gripper_right_eef_rot_axis_angle", "gripper_right_gripper_width")}
        assert len(grid) == ends[k + 1] - ends[k]
        T_true, w_true = _truth_tcp(ep / "raw", grid, rig.cameras["ultrawide"].T_imu_cam, rig.T_cam_tcp)
        T_est = T_from(e["gripper_right_eef_pos"], e["gripper_right_eef_rot_axis_angle"])
        rel_t = np.linalg.inv(T_true[0]) @ T_true
        rel_e = np.linalg.inv(T_est[0]) @ T_est
        dpos = np.linalg.norm(rel_t[:, :3, 3] - rel_e[:, :3, 3], axis=1)
        drot = np.degrees(Rotation.from_matrix(np.swapaxes(rel_t[:, :3, :3], 1, 2) @ rel_e[:, :3, :3]).magnitude())
        assert dpos.max() < 1e-3, dpos.max()
        assert drot.max() < 0.1, drot.max()
        # step-to-step too
        st_t = np.linalg.inv(T_true[:-1]) @ T_true[1:]
        st_e = np.linalg.inv(T_est[:-1]) @ T_est[1:]
        assert np.abs(st_t[:, :3, 3] - st_e[:, :3, 3]).max() < 1e-3
        dw = np.abs(e["gripper_right_gripper_width"][:, 0] - (w_true - gr.min_width))
        assert dw.max() < 2e-3, dw.max()
        assert np.isclose(gr.min_width, 0.030, atol=2e-3)


def test_raw_untouched_and_no_extra_folders(tmp_path_factory, tmp_path):
    root = copy_root(template_root(tmp_path_factory), tmp_path / "root")
    td = task_dir(root)
    (td / "episode_9").mkdir()                       # an interrupted intake's empty episode dir
    (td / ".task.yaml.tmp-123").write_text("x")      # leftovers that must be ignored
    raws = {p: tree_digest(p) for p in list(td.glob("episode_*/raw")) + list(td.glob("calibration/*/raw"))}
    before = {p.relative_to(root.path) for p in root.path.rglob("*") if p.is_dir()}
    res = run_process(root, TASK, slam="oracle", log=quiet)
    assert res.exit_code == 0
    assert [e["slot"] for e in res.report["episodes"]] == [f"episode_{k}" for k in range(N_DEMOS)]
    assert {p: tree_digest(p) for p in raws} == raws
    assert not (td / "episode_9" / "processed").exists()
    new_dirs = {p.relative_to(root.path) for p in root.path.rglob("*") if p.is_dir()} - before
    for d in new_dirs:
        parts = d.parts
        assert "processed" in parts or parts[:3] == ("tasks", TASK, "output"), d       # output/: report.json only
    assert sorted(p.name for p in (td / "output").iterdir()) == ["report.json"]


def test_status_command(done, monkeypatch, capsys):
    root, _ = done
    monkeypatch.setenv("DATAPROC_ROOT", str(root.path))
    assert cli.main(["status", TASK]) == 0
    out = capsys.readouterr().out
    assert "calibration: done (cal_episode_000000)" in out
    assert "3 exported" in out and "not episodes" not in out
    for k in range(N_DEMOS):
        assert re.search(rf"^episode_{k}\s+demo_episode_00000{k}\s+export\s+exported\s+150\s+-$", out, re.M), out


def test_status_lists_staging_and_failed(tmp_path_factory, tmp_path, monkeypatch, capsys):
    """``.staging/<x>`` (waiting for SLAM, or in progress) and ``failed/<x>`` (its reason.json) are listed apart
    from the episodes; a missing folder is an error."""
    import shutil

    root = copy_root(processed_root(tmp_path_factory)[0], tmp_path / "root")
    td = task_dir(root)
    monkeypatch.setenv("DATAPROC_ROOT", str(root.path))
    (td / ".staging").mkdir()
    shutil.copytree(td / "episode_0", td / ".staging" / "s_wait")
    p = td / ".staging" / "s_wait" / "processed" / "status.json"
    st = json.loads(p.read_text())
    st["slam"] = {"status": "failed", "reasons": ["slam_unavailable"], "reason": "no SLAM backend installed"}
    for s in STAGES[2:]:
        st[s] = {"status": "skipped", "reason": "slam failed"}
    p.write_text(json.dumps(st))
    (td / "failed" / "s_bad" / "raw").mkdir(parents=True)
    (td / "failed" / "s_bad" / "reason.json").write_text(json.dumps({"stage": "validate", "reasons": ["imu_gap"]}))
    assert cli.main(["status", TASK]) == 0
    out = capsys.readouterr().out
    assert "3 episodes, 3 exported" in out and "not episodes (never packaged):" in out
    assert re.search(r"^\.staging/s_wait\s+slam\s+waiting for SLAM\s+slam: slam_unavailable$", out, re.M), out
    assert re.search(r"^failed/s_bad\s+validate\s+failed\s+imu_gap$", out, re.M), out
    assert cli.main(["status", "nope"]) == 1
    assert "dataproc calibrate first" in capsys.readouterr().err


def test_status_before_processing(tmp_path_factory, tmp_path, monkeypatch, capsys):
    root = copy_root(template_root(tmp_path_factory), tmp_path / "root")
    monkeypatch.setenv("DATAPROC_ROOT", str(root.path))
    before = tree_digest(root.path)
    assert cli.main(["status", TASK]) == 0
    out = capsys.readouterr().out
    assert "pending" in out and "calibration: not_processed" in out
    assert tree_digest(root.path) == before            # read-only


def test_cli_process_rerun_prints_table(tmp_path_factory, tmp_path, monkeypatch, capsys):
    root = copy_root(processed_root(tmp_path_factory)[0], tmp_path / "root")
    monkeypatch.setenv("DATAPROC_ROOT", str(root.path))
    from dataproc.paths import data_root
    from dataproc.process import cli_process

    assert cli_process(data_root(), TASK, slam="oracle") == 0      # dataproc process is no longer a CLI command
    out = capsys.readouterr().out
    assert "ran: nothing (up to date)" in out and "exported" in out and "report.json" in out
    for k in range(N_DEMOS):
        st = load_status(task_dir(root) / f"episode_{k}" / "processed")
        assert set(st) == set(STAGES)
