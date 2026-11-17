"""umiproc.viz.slam_run (``umiproc plot-run``): the numbers on a tiny fork-like CSV + log, and that a PNG is written."""

import json
from pathlib import Path

import numpy as np
import pytest

from umiproc.slam.log import parse_log
from umiproc.slam.trajectory_csv import CSV_HEADER, read_fork_csv
from umiproc.viz import slam_run as V

T0 = 1_000_000_000_000
DT = 33_333_000                   # a whole number of us, so the CSV's 6-decimal column is exact

# (state, is_lost, keyframe, position): saved map 1, 2, 3, 5; later map 6, 7, 9; not tracking 0, 4 (IMU-only), 8
ROWS = [(1, True, False, None), (2, False, True, (0, 0, 0)), (2, False, False, (1, 0, 0)), (2, False, True, (1, 2, 0)),
        (3, False, False, (1, 2, 1)), (2, False, False, (1, 2, 2)), (2, True, False, None), (2, True, False, None),
        (4, True, False, None), (2, True, False, None)]
SUMMARY = {"format": "umiproc.slam_summary.v2", "exit_code": 0, "error": None, "frames_total": 10, "frames_fed": 10,
           "lost_frames": 6, "not_tracking_frames": 2, "stopped_early": False, "stop_reason": None,
           "state_counts": {"1": 1, "2": 7, "3": 1, "4": 1},
           "maps": [{"id": 0, "keyframes": 3, "map_points": 50, "imu_initialized": False, "inertial_ba1": False,
                     "inertial_ba2": False},
                    {"id": 1, "keyframes": 7, "map_points": 90, "imu_initialized": True, "inertial_ba1": True,
                     "inertial_ba2": True}],
           "t0_ns": T0, "map_removed": False, "options": {"save_map": "map_atlas.osa"},
           "pacing": {"track_ms": {"median": 30.0, "p95": 40.0, "max": 50.0}, "frames_over_period": 1,
                      "feed_rate_hz": 30.0, "recording_rate_hz": 1e9 / DT, "feed_wall_s": 0.3}}


def _write(tmp_path: Path) -> tuple[Path, Path]:
    lines, prev = [",".join(CSV_HEADER)], "0.000000"
    for k, (state, lost, kf, pos) in enumerate(ROWS):
        t = prev if lost else f"{k * DT / 1e9:.6f}"   # lost rows repeat the previous time, as the fork writes them
        prev = t
        pose = (0, 0, 0, 0, 0, 0, 0) if lost else (*pos, 0.0, 0.0, 0.0, 1.0)
        lines.append(",".join(str(v) for v in (k, t, state, str(lost).lower(), str(kf).lower(), *pose)))
    csv_path = tmp_path / "trajectory.csv"
    csv_path.write_text("\n".join(lines) + "\n")
    Path(str(csv_path) + ".t0_ns").write_text(f"{T0}\n")
    log_path = tmp_path / "log.txt"
    log_path.write_text("Creation of new map with id: 0\nUMIPROC_SUMMARY " + json.dumps(SUMMARY) + "\n")
    return csv_path, log_path


def test_stats(tmp_path):
    csv_path, log_path = _write(tmp_path)
    times = T0 + DT * np.arange(10, dtype=np.int64)
    traj = read_fork_csv(csv_path, times, frames_fed=10)
    assert V.classify(traj).tolist() == [0, 2, 2, 2, 0, 2, 1, 1, 0, 1]
    st = V.compute_stats(traj)
    assert (st.frames, st.n_saved, st.n_later, st.n_not_tracking) == (10, 4, 3, 3)
    assert st.saved_span_s == pytest.approx((DT / 1e9, 5 * DT / 1e9))
    assert st.longest_s[:2] == (1, 3) and st.longest_s[2:] == pytest.approx((DT / 1e9, 3 * DT / 1e9))
    assert st.extent_m == pytest.approx((1.0, 2.0, 2.0))
    assert st.path_length_m == pytest.approx(5.0)            # 1 + 2 + 2 over frames 1, 2, 3, 5
    assert st.keyframes_marked == 2
    summary = parse_log(log_path.read_text()).summary
    assert V.saved_map(summary).id == 1                       # the biggest map is the CSV's
    text = "\n".join(V.stats_lines(st, summary))
    assert "4 (40.0%)" in text and "length 5.00 m" in text


def test_times_from_csv_column(tmp_path):
    csv_path, _ = _write(tmp_path)
    t = V.times_from_csv_column(csv_path, T0, 10, DT)
    assert t.tolist() == (T0 + DT * np.arange(10)).tolist()


def test_command_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from umiproc.cli import main

    csv_path, log_path = _write(tmp_path)
    out = tmp_path / "fig.png"
    assert main(["plot-run", "--csv", str(csv_path), "--log", str(log_path), "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
