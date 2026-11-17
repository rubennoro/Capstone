"""Read the fork's trajectory CSV (System::SaveTrajectoryCSV) against the frames that were fed.

``frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w``, one row per frame fed to
LocalizeMonocular: **row k is line k of times.txt** (Tracking::Track pushes exactly one row per call). The rows are
joined on frame_idx and the frame times come from times.txt, never from the timestamp column: before the first
tracked frame that column is garbage (``.back()`` of an empty list), lost frames repeat the previous time, and it has
6 decimals. It is only used as a misalignment guard on posed rows. ``OUT.csv.t0_ns`` (written by umiproc_slam) is
the column's origin and must be times.txt's first line.

Pose: T_W_cam (the ultrawide camera's optical frame in W), W = the first keyframe of the biggest map; translation
x,y,z and quaternion q_x,q_y,q_z,q_w.

Which rows count:
  posed    is_lost false with a valid pose (finite, unit quaternion) that is not a re-pushed copy of the previous
           row (``stale``: same timestamp as the previous posed row). Includes RECENTLY_LOST (state 3) IMU-only
           predictions.
  tracked  posed and state == OK (2): visually tracked. umiproc's "tracked" everywhere.
A map reset marks its frames is_lost=true without changing their state, so state 2 alone does not mean tracked.

A header-only file (patch 0002: no keyframe in any map at the end) means every fed frame is lost: it is read as
that, with state -1 and ``header_only`` set.

Refused (TrajectoryCsvError): a header other than the fork's, row count other than the frames fed, frame_idx not
0..N-1, a t0_ns other than times.txt's first line, unknown states, malformed values, and posed rows whose timestamp
disagrees with times.txt by more than ``TIMESTAMP_TOLERANCE_NS`` (rows and frames are not aligned).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from umiproc.slam.log import STATE_NAMES, STATE_OK

CSV_HEADER = ("frame_idx", "timestamp", "state", "is_lost", "is_keyframe", "x", "y", "z", "q_x", "q_y", "q_z", "q_w")
STATE_RECENTLY_LOST = 3
TIMESTAMP_TOLERANCE_NS = 1_000_000      # the column has 1 us resolution; 1 ms apart means misaligned rows
QUATERNION_NORM_TOLERANCE = 1e-3


class TrajectoryCsvError(ValueError):
    """The CSV does not match the frames that were fed (or is malformed)."""


@dataclass(frozen=True)
class ForkTrajectory:
    """The fork's per-frame output, one entry per fed frame (row k = times.txt line k)."""

    path: Path
    t0_ns: int
    t_ns: np.ndarray             # (N,) int64 frame times from times.txt
    state: np.ndarray            # (N,) int64 Tracking::eTrackingState after the frame
    is_lost: np.ndarray          # (N,) bool, as written
    is_keyframe: np.ndarray      # (N,) bool, as written (unreliable after a relocalization re-shifted keyframe times)
    posed: np.ndarray            # (N,) bool: has a usable pose (tracked or an IMU-only prediction)
    tracked: np.ndarray          # (N,) bool: posed and state == OK
    T_W_cam: np.ndarray          # (N, 4, 4) float64, NaN where not posed
    n_frames_total: int          # lines in times.txt (> N when the run stopped early)
    anomalies: dict              # rows marked not posed although is_lost is false: {"stale": n, "bad_pose": n}
    header_only: bool = False    # no rows: no map had a keyframe, every fed frame lost (state -1)

    @property
    def n_frames(self) -> int:
        return int(len(self.t_ns))

    @property
    def n_tracked(self) -> int:
        return int(self.tracked.sum())

    @property
    def lost(self) -> np.ndarray:
        """Not visually tracked (the complement of ``tracked``)."""
        return ~self.tracked

    @property
    def first_tracked_index(self) -> int | None:
        idx = np.flatnonzero(self.tracked)
        return int(idx[0]) if len(idx) else None

    @property
    def tracked_fraction(self) -> float:
        return float(self.tracked.mean()) if self.n_frames else 0.0

    def state_counts(self) -> dict[str, int]:
        names, counts = np.unique(self.state, return_counts=True)
        return {STATE_NAMES[int(s)]: int(c) for s, c in zip(names, counts)}


def read_times(path: str | Path) -> np.ndarray:
    """times.txt as umiproc_slam reads it: one integer ns per line, blank and '#' lines skipped."""
    out = []
    for k, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            out.append(int(line))
        except ValueError as e:
            raise TrajectoryCsvError(f"{path}: line {k} is not an integer time: {line!r}") from e
    return np.asarray(out, dtype=np.int64)


def _bool_cell(v: str, where: str) -> bool:
    v = v.strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    raise TrajectoryCsvError(f"{where}: not true/false: {v!r}")


def read_fork_csv(csv_path: str | Path, times: str | Path | np.ndarray, *, t0_ns: int | None = None,
                  frames_fed: int | None = None) -> ForkTrajectory:
    """Read ``csv_path`` aligned to ``times`` (times.txt path or array of the frame times in ns).

    ``t0_ns``: the timestamp column's origin (default: ``csv_path + ".t0_ns"``). ``frames_fed``: frames the run fed
    (the summary's ``frames_fed``; default all of times.txt); the CSV must have exactly that many rows.
    """
    csv_path = Path(csv_path)
    t_all = read_times(times) if isinstance(times, (str, Path)) else np.asarray(times, dtype=np.int64)
    if t_all.ndim != 1 or len(t_all) == 0:
        raise TrajectoryCsvError("no frame times")
    if t0_ns is None:
        t0_path = Path(str(csv_path) + ".t0_ns")
        try:
            t0_ns = int(t0_path.read_text().split()[0])
        except (OSError, IndexError, ValueError) as e:
            raise TrajectoryCsvError(f"cannot read the time origin {t0_path}: {e}") from e
    if t0_ns != int(t_all[0]):
        raise TrajectoryCsvError(f"{csv_path}: t0_ns {t0_ns} is not the first frame time {int(t_all[0])} "
                                 "(trajectory from another sequence?)")
    n_fed = len(t_all) if frames_fed is None else int(frames_fed)
    if not 0 <= n_fed <= len(t_all):
        raise TrajectoryCsvError(f"frames_fed {n_fed} is outside 0..{len(t_all)} (the frames in times.txt)")

    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            rows = [r for r in reader if r]
    except OSError as e:
        raise TrajectoryCsvError(f"cannot read {csv_path}: {e}") from e
    if header is None or tuple(h.strip() for h in header) != CSV_HEADER:
        raise TrajectoryCsvError(f"{csv_path}: header {header!r} is not the fork's {','.join(CSV_HEADER)}")
    n = len(rows)
    if n == 0 and n_fed > 0:                  # patch 0002: no keyframe in any map, every frame lost
        return ForkTrajectory(path=csv_path, t0_ns=int(t0_ns), t_ns=t_all[:n_fed], state=np.full(n_fed, -1, np.int64),
                              is_lost=np.ones(n_fed, bool), is_keyframe=np.zeros(n_fed, bool),
                              posed=np.zeros(n_fed, bool), tracked=np.zeros(n_fed, bool),
                              T_W_cam=np.full((n_fed, 4, 4), np.nan), n_frames_total=int(len(t_all)),
                              anomalies={"stale": 0, "bad_pose": 0}, header_only=True)
    if n != n_fed:
        raise TrajectoryCsvError(f"{csv_path}: {n} rows but {n_fed} frames were fed (times.txt has {len(t_all)}): "
                                 "row k must be fed frame k")

    frame_idx = np.empty(n, np.int64)
    state = np.empty(n, np.int64)
    is_lost = np.empty(n, bool)
    is_kf = np.empty(n, bool)
    pose = np.empty((n, 7), np.float64)
    ts: list[str] = []
    for k, r in enumerate(rows):
        where = f"{csv_path}: row {k + 1}"
        if len(r) != len(CSV_HEADER):
            raise TrajectoryCsvError(f"{where}: {len(r)} columns, want {len(CSV_HEADER)}")
        try:
            frame_idx[k] = int(r[0])
            state[k] = int(r[2])
            pose[k] = [float(v) for v in r[5:12]]
        except ValueError as e:
            raise TrajectoryCsvError(f"{where}: malformed number: {e}") from e
        if int(state[k]) not in STATE_NAMES:
            raise TrajectoryCsvError(f"{where}: unknown tracking state {int(state[k])}")
        is_lost[k] = _bool_cell(r[3], where)
        is_kf[k] = _bool_cell(r[4], where)
        ts.append(r[1].strip())
    if not np.array_equal(frame_idx, np.arange(n)):
        bad = int(np.flatnonzero(frame_idx != np.arange(n))[0])
        raise TrajectoryCsvError(f"{csv_path}: frame_idx is not 0..{n - 1} (row {bad + 1} has {int(frame_idx[bad])})")

    t_ns = t_all[:n]
    q_norm = np.linalg.norm(pose[:, 3:], axis=1)
    valid = np.isfinite(pose).all(axis=1) & (np.abs(q_norm - 1.0) <= QUATERNION_NORM_TOLERANCE)
    candidate = ~is_lost & valid
    stale = np.zeros(n, bool)
    prev_ts = None
    for k in range(n):
        if candidate[k]:
            stale[k] = ts[k] == prev_ts
            prev_ts = ts[k]
    posed = candidate & ~stale
    tracked = posed & (state == STATE_OK)

    worst = 0
    for k in np.flatnonzero(posed):           # misalignment guard: on posed rows the column is the fed time
        try:
            t = t0_ns + int((Decimal(ts[k]) * 1_000_000_000).to_integral_value())
        except InvalidOperation as e:
            raise TrajectoryCsvError(f"{csv_path}: row {k + 1}: timestamp {ts[k]!r} is not a number") from e
        worst = max(worst, abs(t - int(t_ns[k])))
        if worst > TIMESTAMP_TOLERANCE_NS:
            raise TrajectoryCsvError(f"{csv_path}: row {k + 1} (posed) has time {ts[k]} s, but frame {k} of "
                                     f"times.txt is {(int(t_ns[k]) - t0_ns) / 1e9:.6f} s: rows and frames are not "
                                     "aligned")

    T = np.full((n, 4, 4), np.nan)
    if posed.any():
        q = pose[posed, 3:] / q_norm[posed, None]
        Tp = np.tile(np.eye(4), (int(posed.sum()), 1, 1))
        Tp[:, :3, :3] = Rotation.from_quat(q).as_matrix()          # scipy: (x, y, z, w)
        Tp[:, :3, 3] = pose[posed, :3]
        T[posed] = Tp
    return ForkTrajectory(path=csv_path, t0_ns=int(t0_ns), t_ns=t_ns, state=state, is_lost=is_lost,
                          is_keyframe=is_kf, posed=posed, tracked=tracked, T_W_cam=T, n_frames_total=int(len(t_all)),
                          anomalies={"stale": int(stale.sum()), "bad_pose": int((~is_lost & ~valid).sum())})
