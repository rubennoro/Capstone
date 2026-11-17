"""umiproc.slam.trajectory_csv: the fork's CSV joined to times.txt, on a recorded excerpt and fork-like synthetic files
(including the ways a file can be misaligned)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from umiproc.slam.trajectory_csv import CSV_HEADER, TrajectoryCsvError, read_fork_csv, read_times

FIX = Path(__file__).parent / "fixtures" / "slam"
T0 = 1_000_000_000_000
DT = 33_366_700                   # ~30 fps


def _times(n: int) -> np.ndarray:
    return T0 + DT * np.arange(n, dtype=np.int64)


def _write(tmp_path: Path, rows: list[list], *, t0: int | None = T0, header=CSV_HEADER, name="t.csv") -> Path:
    p = tmp_path / name
    p.write_text(",".join(header) + "\n" + "".join(",".join(str(v) for v in r) + "\n" for r in rows))
    if t0 is not None:
        Path(str(p) + ".t0_ns").write_text(f"{t0}\n")
    return p


def _row(k: int, t_s: float, state: int, lost: bool, pose=(0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0), kf=False) -> list:
    """One row as SaveTrajectoryCSV writes it (lost rows: zeros)."""
    if lost:
        pose = (0, 0, 0, 0, 0, 0, 0)
    return [k, f"{t_s:.6f}", state, "true" if lost else "false", "true" if kf else "false", *pose]


def _fork_like(n: int, t: np.ndarray) -> list[list]:
    """Frames 0-2 waiting to relocalize (garbage time, state 6), 3-6 tracked, 7 RECENTLY_LOST (IMU-only), 8 OK,
    9 lost again (repeats the previous time), the rest tracked."""
    secs = (t - t[0]) / 1e9
    rows = []
    last = 1.23e-300                       # what .back() of an empty list gives
    for k in range(n):
        if k <= 2 or k == 9:
            rows.append(_row(k, last, 6, True))
        elif k == 7:
            rows.append(_row(k, secs[k], 3, False))
            last = secs[k]
        else:
            rows.append(_row(k, secs[k], 2, False, kf=(k == 3)))
            last = secs[k]
    return rows


# ---------------------------------------------------------------------------------------------- recorded excerpt

def test_recorded_excerpt():
    tr = read_fork_csv(FIX / "localize_first40.csv", FIX / "localize_first40_times.txt")
    times = read_times(FIX / "localize_first40_times.txt")
    assert tr.n_frames == 40 == tr.n_tracked and tr.first_tracked_index == 0
    assert np.array_equal(tr.t_ns, times) and tr.t0_ns == int(times[0])
    assert tr.T_W_cam.shape == (40, 4, 4)
    R = tr.T_W_cam[:, :3, :3]
    assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-6)
    first = (FIX / "localize_first40.csv").read_text().splitlines()[1].split(",")
    assert np.allclose(tr.T_W_cam[0, :3, 3], [float(v) for v in first[5:8]])
    assert np.allclose(R[0], Rotation.from_quat([float(v) for v in first[8:12]]).as_matrix(), atol=1e-6)
    assert tr.state_counts() == {"OK": 40}


# ---------------------------------------------------------------------------------------------- what counts

def test_tracked_posed_and_lost(tmp_path):
    t = _times(14)
    tr = read_fork_csv(_write(tmp_path, _fork_like(14, t)), t)
    assert tr.tracked.tolist() == [False] * 3 + [True] * 4 + [False, True, False] + [True] * 4
    assert tr.posed.tolist() == [False] * 3 + [True] * 6 + [False] + [True] * 4      # RECENTLY_LOST is posed
    assert np.isnan(tr.T_W_cam[0]).all() and np.isfinite(tr.T_W_cam[7]).all()
    assert tr.lost.sum() == 14 - tr.n_tracked
    assert tr.is_keyframe[3] and tr.first_tracked_index == 3
    assert tr.anomalies == {"stale": 0, "bad_pose": 0} and not tr.header_only
    assert tr.state_counts() == {"OK": 9, "RECENTLY_LOST": 1, "INIT_RELOCALIZE": 4}


def test_reset_rows_are_not_tracked(tmp_path):
    # a map reset marks its frames lost (is_lost true, zero pose) but keeps their state 2
    t = _times(4)
    secs = (t - t[0]) / 1e9
    rows = [_row(0, secs[0], 2, True), _row(1, secs[1], 2, True), _row(2, secs[2], 2, False),
            _row(3, secs[3], 2, False)]
    tr = read_fork_csv(_write(tmp_path, rows), t)
    assert tr.tracked.tolist() == [False, False, True, True]


def test_stale_and_bad_pose_rows(tmp_path):
    t = _times(5)
    secs = (t - t[0]) / 1e9
    rows = [_row(0, secs[0], 2, False), _row(1, secs[0], 2, False),          # re-pushed copy of row 0
            _row(2, secs[2], 2, False, pose=(0, 0, 0, 0, 0, 0, 0)),         # not lost but no rotation
            _row(3, secs[3], 2, False, pose=(0, 0, 0, 0, 0, 0, 2.0)),       # not a unit quaternion
            _row(4, secs[4], 2, False)]
    tr = read_fork_csv(_write(tmp_path, rows), t)
    assert tr.tracked.tolist() == [True, False, False, False, True]
    assert tr.anomalies == {"stale": 1, "bad_pose": 2}


def test_stopped_early(tmp_path):
    t = _times(20)
    tr = read_fork_csv(_write(tmp_path, _fork_like(12, t)), t, frames_fed=12)
    assert tr.n_frames == 12 and tr.n_frames_total == 20 and np.array_equal(tr.t_ns, t[:12])


def test_header_only_means_every_frame_lost(tmp_path):
    t = _times(6)
    tr = read_fork_csv(_write(tmp_path, []), t)
    assert tr.header_only and tr.n_frames == 6 and tr.n_tracked == 0 and tr.first_tracked_index is None
    assert (tr.state == -1).all() and tr.tracked_fraction == 0.0


def test_t0_argument_and_times_file(tmp_path):
    t = _times(14)
    (tmp_path / "times.txt").write_text("# t_ns\n" + "\n".join(str(v) for v in t) + "\n\n")
    p = _write(tmp_path, _fork_like(14, t), t0=None)
    tr = read_fork_csv(p, tmp_path / "times.txt", t0_ns=int(t[0]))
    assert tr.n_tracked == 9


# ---------------------------------------------------------------------------------------------- refused files

def test_row_count_mismatch(tmp_path):
    t = _times(14)
    with pytest.raises(TrajectoryCsvError, match="13 rows but 14 frames"):
        read_fork_csv(_write(tmp_path, _fork_like(13, t)), t)
    with pytest.raises(TrajectoryCsvError, match="frames_fed"):
        read_fork_csv(_write(tmp_path, _fork_like(14, t)), t, frames_fed=15)


def test_misaligned_rows(tmp_path):
    # the CSV of a run fed from frame 1 on, read against times starting at frame 0: same t0 file, wrong rows
    t = _times(15)
    rows = _fork_like(14, t[1:])
    for r in rows:
        if r[3] == "false":
            r[1] = f"{float(r[1]) + DT / 1e9:.6f}"
    with pytest.raises(TrajectoryCsvError, match="not aligned"):
        read_fork_csv(_write(tmp_path, rows), t[:14])


def test_t0_mismatch(tmp_path):
    t = _times(14)
    with pytest.raises(TrajectoryCsvError, match="t0_ns"):
        read_fork_csv(_write(tmp_path, _fork_like(14, t), t0=T0 + 1), t)
    p = _write(tmp_path, _fork_like(14, t), t0=None, name="no_t0.csv")
    with pytest.raises(TrajectoryCsvError, match="time origin"):
        read_fork_csv(p, t)


def test_frame_idx_not_sequential(tmp_path):
    t = _times(14)
    rows = _fork_like(14, t)
    rows[5][0] = 6
    with pytest.raises(TrajectoryCsvError, match="frame_idx"):
        read_fork_csv(_write(tmp_path, rows), t)


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows[4].__setitem__(2, 9), "unknown tracking state"),
    (lambda rows: rows[4].__setitem__(3, "maybe"), "true/false"),
    (lambda rows: rows[4].__setitem__(5, "x"), "malformed number"),
    (lambda rows: rows[4].pop(), "columns"),
    (lambda rows: rows[4].__setitem__(1, "abc"), "not a number"),
])
def test_malformed_rows(tmp_path, mutate, match):
    t = _times(14)
    rows = _fork_like(14, t)
    mutate(rows)
    with pytest.raises(TrajectoryCsvError, match=match):
        read_fork_csv(_write(tmp_path, rows), t)


def test_wrong_header(tmp_path):
    t = _times(14)
    with pytest.raises(TrajectoryCsvError, match="header"):
        read_fork_csv(_write(tmp_path, _fork_like(14, t), header=("frame", "t") + CSV_HEADER[2:]), t)


def test_bad_times_file(tmp_path):
    (tmp_path / "times.txt").write_text("1\nnope\n")
    with pytest.raises(TrajectoryCsvError, match="not an integer"):
        read_times(tmp_path / "times.txt")
