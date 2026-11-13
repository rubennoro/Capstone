"""``calibration_diagnostics`` and the failure hints, on synthetic tag detections (no rendering)."""

from __future__ import annotations

import numpy as np

from dataproc.calibrate.diagnostics import format_failure, hints
from dataproc.calibrate.task import CalibrateResult, source_problem_code
from dataproc.sim.episode import DEFAULT_RIG_PATH, fast_rig
from dataproc.stages.calibration import CalibrationFit
from dataproc.stages.width import calibration_diagnostics, count_open_close_cycles

_, RIG = fast_rig(DEFAULT_RIG_PATH)
Z = RIG.aruco.nominal_z_m


def _frame(width, z=Z, left=True, right=True):
    f = {}
    if left:
        f[0] = {"tvec": [-width / 2, 0.01, z]}
    if right:
        f[1] = {"tvec": [width / 2, 0.01, z]}
    return f


def _sweep(n_cycles, n_per=20):
    w = 0.03 + 0.03 * (1 - np.cos(np.linspace(0, 2 * np.pi * n_cycles, n_cycles * n_per + 1)))
    return [float(x) for x in w]


def test_cycles_counted_with_hysteresis():
    assert count_open_close_cycles(_sweep(5)) == 5
    assert count_open_close_cycles([0.03, 0.09] * 3) == 3
    assert count_open_close_cycles([0.05] * 10) == 0
    assert count_open_close_cycles([]) == 0
    jitter = [0.03, 0.031, 0.029, 0.03, 0.09, 0.089, 0.091, 0.03]     # jitter at one level does not count
    assert count_open_close_cycles(jitter) == 1


def test_diagnostics_counts():
    frames = [_frame(w) for w in _sweep(5)]                            # 101 both-tag frames in the window
    frames += [_frame(0.05, left=False)] * 9 + [{}] * 10 + [_frame(0.05, z=0.5)] * 5
    d = calibration_diagnostics(frames, RIG)
    assert d["n_frames"] == 125
    assert (d["n_left"], d["n_right"], d["n_both"], d["n_in_window"]) == (106, 115, 106, 101)
    assert d["detection_rate"] == 106 / 125 and d["n_cycles"] == 5
    assert abs(d["closed_width_m"] - 0.03) < 1e-9 and abs(d["open_width_m"] - 0.09) < 1e-9
    assert abs(d["open_distance_m"] - 0.09) < 1e-9
    assert d["z_window_m"] == [Z - 0.08, Z + 0.08] and d["median_z_m"] == Z
    empty = calibration_diagnostics([{}] * 4, RIG)
    assert empty["median_z_m"] is None and empty["closed_width_m"] is None and empty["n_cycles"] == 0


def _res(reasons, diag, message="x", **details):
    fit = CalibrationFit(reasons=reasons, reason=message, diagnostics=diag, details=details)
    return CalibrateResult(False, "block_raise", "episode_000001", None, "rig_v1_fast", fit=fit, reasons=reasons,
                           message=message)


def test_low_detection_rate_hint():
    frames = [_frame(0.05)] * 10 + [{}] * 90
    res = _res(["gripper_range_fit"], calibration_diagnostics(frames, RIG), "detection rate 0.100 < 0.2 threshold")
    (h,) = hints(res, RIG)
    assert "seen in only 10% of frames" in h and "lighting" in h and "blur" in h
    text = format_failure(res, RIG)
    assert "tag 0 seen 10 (10%), tag 1 seen 10 (10%), both 10 (10%; fit needs 20%" in text
    assert "dataproc calibrate block_raise episode_000001` again" in text


def test_rig_hint():
    res = _res(["rig_resolution_mismatch", "rig_fps_mismatch"], {})
    assert len(hints(res, RIG)) == 1 and "wrong --rig rig_v1_fast" in hints(res, RIG)[0]


def test_source_problem_codes():
    assert source_problem_code("not a directory") == "source_missing"
    assert source_problem_code("no COMPLETE (still being recorded or transferred)") == "still_syncing"
    assert source_problem_code("no MANIFEST.sha256") == "manifest_failed"
    assert source_problem_code("manifest check failed: x") == "manifest_failed"
    assert source_problem_code("the Pi marked it invalid (meta.json valid: false)") == "pi_invalid"
    assert source_problem_code("the Pi reported problems: [1]") == "pi_invalid"
    assert source_problem_code("meta.json unreadable (x)") == "meta_unreadable"
