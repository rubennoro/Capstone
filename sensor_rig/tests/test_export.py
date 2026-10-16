"""Export on hand-built raw episodes with a known trajectory (WP-A definition of done)."""

import json

import numpy as np
import pytest

from sensor_rig.cli import main as cli_main
from sensor_rig.export.exporter import ExportOptions, export_episode
from sensor_rig.export.validate import validate_iphone_episode
from sensor_rig.export.video import probe_video
from sensor_rig.timefmt import iso_ms, parse_iso
from wpa_helpers import MONO_BASE, START_ISO, expected_arkit, make_raw_episode

WALL = parse_iso(START_ISO) - MONO_BASE  # unix = corrected mono + WALL


def _load(out_dir):
    return json.loads((out_dir / f"{out_dir.name}.json").read_text())


def _rel_err(Ta, Tb):
    """max rotation (rad) / translation (m) error between the relative motions of two pose sequences."""
    Ra = np.linalg.inv(Ta[0]) @ Ta
    Rb = np.linalg.inv(Tb[0]) @ Tb
    D = np.linalg.inv(Ra) @ Rb
    ang = np.arccos(np.clip((np.trace(D[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1))
    return ang.max(), np.abs(Ra[:, :3, 3] - Rb[:, :3, 3]).max()


def test_native_export_matches_truth(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=60)
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "exported" and r.grid_mode == "native" and r.frame_count == 150
    j = _load(r.out_dir)
    # grid = true main capture times (raw stamp + offset); strings via the record-start wall offset
    t_cap = MONO_BASE + 20_000_000 + np.round(np.arange(150) * 1e9 / 60).astype(np.int64)
    assert j["poseTimes"] == [iso_ms(t + WALL) for t in t_cap]
    T = np.asarray(j["poseTransforms"], dtype=np.float64)
    want = expected_arkit(test_rig, t_cap)
    assert np.abs(T - want).max() < 2e-6  # float32 rounding only (trajectory keys = grid times)
    assert np.all(np.linalg.norm(T[:, :3, 3] - [1.0, -0.2, 5.0], axis=1) < 1.0)  # world offset applied
    # ultrawide: ~10 Hz, slotted to main ticks, every non-empty string is the slot's main time
    occ = [i for i, s in enumerate(j["ultrawideRGBTimes"]) if s]
    assert 24 <= len(occ) <= 26 and set(np.diff(occ)) <= {5, 6, 7}
    assert all(j["ultrawideRGBTimes"][i] == j["poseTimes"][i] for i in occ)
    p = probe_video(r.out_dir / f"{r.out_dir.name}_ultrawidergb.mp4")
    rel = [round((t_cap[i] - t_cap[occ[0]]) * 600 / 1e9) for i in occ]
    assert p["pts_list"] == rel and p["frames_header"] == len(occ)
    assert j["mainCameraIntrinsics"] == test_rig.virtual_camera("main", "export").K_float32_rows()
    assert j["deviceIdentifier"] == "JetsonRig1,1" and j["gripperCalibrationRunName"].endswith("_left")


def test_fill_grid_and_sparse_trajectory(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, main_fps=30, traj_rate_hz=20)
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "exported" and r.grid_mode == "fill", r
    j = _load(r.out_dir)
    t = np.array([parse_iso(s) for s in j["poseTimes"]]) - WALL
    assert np.all(np.abs(np.diff(t) - 1e9 / 60) <= 1e6)  # 60 Hz grid (ms strings)
    T = np.asarray(j["poseTransforms"], dtype=np.float64)
    t_grid = MONO_BASE + 20_000_000 + np.round(np.arange(len(T)) * 1e9 / 60).astype(np.int64)
    rot, trans = _rel_err(T, expected_arkit(test_rig, t_grid))
    assert rot < 2e-3 and trans < 2e-3, (rot, trans)  # SLERP/linear between 20 Hz keys
    assert validate_iphone_episode(r.out_dir) == []


def test_head_trim_and_mid_gap_abort(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, traj_start_s=0.5, rid="11111")
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "exported" and 0.45 < r.trim_head_s < 0.55 and r.frame_count < 150
    j = _load(r.out_dir)
    assert parse_iso(j["poseTimes"][0]) - WALL >= MONO_BASE + 20_000_000 + 500_000_000 - 1_000_000

    raw2 = make_raw_episode(tmp_path / "raw", test_rig, traj_gap=(1.0, 0.3), rid="22222")
    r2 = export_episode(raw2, tmp_path / "out")
    assert r2.status == "aborted" and r2.reason == "pose_gap_mid_episode" and r2.out_dir is None
    assert not any(p.name.startswith(raw2.name) for p in (tmp_path / "out").rglob("*"))
    log = (tmp_path / "out" / ".sensor_rig" / "export_log.jsonl").read_text().splitlines()
    assert json.loads(log[-1])["reason"] == "pose_gap_mid_episode"
    assert not list((tmp_path / "out").glob(".partial-*"))


def test_no_trajectory_aborts(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig)
    for p in (raw / "trajectory").iterdir():
        p.unlink()
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "aborted" and r.reason == "no_trajectory"


def test_no_calibration_run_aborts_early(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig, calibration_run="")
    r = export_episode(raw, tmp_path / "out")
    assert r.status == "aborted" and r.reason == "no_calibration_run" and "--calibration-run" in r.warnings[0]
    from wpa_helpers import CALIB_NAME

    r2 = export_episode(raw, tmp_path / "out", ExportOptions(calibration_run=CALIB_NAME))
    assert r2.status == "exported" and _load(r2.out_dir)["gripperCalibrationRunName"] == CALIB_NAME


def test_deterministic_and_skip_existing(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig)
    a = export_episode(raw, tmp_path / "a")
    b = export_episode(raw, tmp_path / "b")
    ja = (a.out_dir / f"{a.name}.json").read_bytes()
    assert ja == (b.out_dir / f"{b.name}.json").read_bytes()
    assert export_episode(raw, tmp_path / "a").status == "skipped"
    assert export_episode(raw, tmp_path / "a", ExportOptions(overwrite=True)).status == "exported"


def test_label_task_and_uw_all(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig)
    r = export_episode(raw, tmp_path / "out", ExportOptions(label_task="block_raise", ultrawide_rate_hz=None))
    j = _load(r.out_dir)
    assert j["labelType"] == "Predefined" and j["taskNames"] == ["block_raise"]
    assert r.uw_count >= 70  # every 30 fps UW frame gets its own slot


def test_refuses_out_root_inside_raw_root(tmp_path, test_rig):
    raw = make_raw_episode(tmp_path / "raw", test_rig)
    r = export_episode(raw, tmp_path / "raw" / "exports")
    assert r.status == "aborted" and r.reason == "bad_out_root"


def test_cli_export_and_validate(tmp_path, test_rig, capsys):
    from wpa_helpers import CALIB_NAME

    make_raw_episode(tmp_path / "raw", test_rig, kind="gripper_calibration", name=CALIB_NAME)
    make_raw_episode(tmp_path / "raw", test_rig)
    assert cli_main(["export", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert out.count("exported") == 2
    assert cli_main(["validate", str(tmp_path / "out")]) == 0
    with pytest.raises(SystemExit):
        cli_main(["export", "--help"])
