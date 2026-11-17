"""UMI's lost-frame rules for one localized demo, converted from UMI's 60 fps to our 30 fps (same times), plus the
relocalization rule we add (docs/DESIGN.md "Checks -> run"). Pure functions of the localization run's per-frame
tracking, so every boundary is testable without the native binary.

UMI (``umi_repo/scripts_slam_pipeline``), for one gripper camera, in the order it applies them:

1. **Stopped** (:data:`MAX_LOST_FRAMES`: UMI 60 at 60 fps = 1 s -> 30): ``03_batch_slam.py:44, 123`` runs the
   localization with ``--max_lost_frames 60``; ``gopro_slam.cc:248-258`` counts the frames ``LocalizeMonocular`` did
   not track (over the whole run, not in a row) and exits at the limit without saving the CSV, so
   ``06_generate_dataset_plan.py:592-596`` skips the demo (no ``camera_trajectory.csv``).
2. **Too many lost** (:data:`MAX_LOST_IN_DEMO`: UMI > 10 at 60 fps = 0.17 s -> > 5): ``06:601-608`` (and
   ``412-415``), ``is_tracked = ~is_lost`` over the demo's frames, more than 10 lost -> skipped.
3. **Too few tracked** (:data:`MIN_TRACKED_FRAMES`: UMI < 60 = 1 s -> < 30): ``06:610-614`` (and ``417-418``).
4. **Split** (:data:`MIN_PIECE_FRAMES`: UMI 24 = 0.4 s -> 12): ``06:629, 719-726`` with ``get_bool_segments``
   (``06:41-54``) cut the tracked frames into runs at every lost frame and drop the runs shorter than
   ``min_episode_length`` (``06:88``); ``06:729-763`` make every remaining run its own episode.

Converted: only the four thresholds (frames at 30 fps for the same time). Kept as UMI does it: the counts are over
every frame of the demo (UMI's window is the whole video for one camera; ``06:555-568`` only shifts it by one frame
to align several cameras, which we do not have), a stopped run drops the demo whatever it tracked, and the split is
at every single lost frame.

Ours (DESIGN "Checks -> run"):

* **tracked** = visually tracked: state OK and not lost (:attr:`umiproc.slam.trajectory_csv.ForkTrajectory.tracked`),
  never an IMU-only prediction (UMI's ``~is_lost`` also counts the fork's RECENTLY_LOST rows). Frames never fed
  (after a stop) count as lost.
* **relocalized within 0.5 s**: the first tracked frame at most :data:`MAX_RELOCALIZE_S` after the first frame (a
  later relocalization is where the fork's inertial link was wrong; map check M3 uses the same limit). Note that
  every frame before it is lost, so the > 5 lost rule already asks for it within the first 6 frames (0.17 s); this
  rule only names the cause.

A drop gives reason codes (:data:`REASONS`), every one that applies; a stopped run gives only ``stopped_lost``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["FPS", "UMI_FPS", "MAX_LOST_FRAMES", "MAX_LOST_IN_DEMO", "MIN_TRACKED_FRAMES", "MIN_PIECE_FRAMES",
           "MAX_RELOCALIZE_S", "LIMITS", "REASONS", "Piece", "RulesResult", "bool_segments", "apply_rules"]

FPS = 30.0                    # the ultrawide's rate (rig_pi_v3)
UMI_FPS = 60.0                # UMI's GoPro
#: 03_batch_slam.py:44 --max_lost_frames 60 at 60 fps (1 s)
MAX_LOST_FRAMES = 30
#: 06_generate_dataset_plan.py:605 "> 10" at 60 fps (0.17 s)
MAX_LOST_IN_DEMO = 5
#: 06_generate_dataset_plan.py:611 "< 60" at 60 fps (1 s)
MIN_TRACKED_FRAMES = 30
#: 06_generate_dataset_plan.py:88 min_episode_length 24 at 60 fps (0.4 s)
MIN_PIECE_FRAMES = 12
#: ours: the first tracked frame within 0.5 s of the first frame (as map check M3)
MAX_RELOCALIZE_S = 0.5
LIMITS = {"max_lost_frames_run": MAX_LOST_FRAMES, "max_lost_in_demo": MAX_LOST_IN_DEMO,
          "min_tracked_frames": MIN_TRACKED_FRAMES, "min_piece_frames": MIN_PIECE_FRAMES,
          "max_relocalize_s": MAX_RELOCALIZE_S, "fps": FPS, "umi_fps": UMI_FPS}
REASONS = {
    "stopped_lost": f"the localization stopped after {MAX_LOST_FRAMES} frames it could not track (UMI: no "
                    "trajectory, the demo is dropped)",
    "too_many_lost": f"more than {MAX_LOST_IN_DEMO} lost frames in the demo (UMI's > 10 at 60 fps)",
    "too_few_tracked": f"fewer than {MIN_TRACKED_FRAMES} tracked frames (UMI's < 60 at 60 fps)",
    "late_relocalization": f"not found in the map within the first {MAX_RELOCALIZE_S} s",
    "no_piece": f"no tracked stretch of at least {MIN_PIECE_FRAMES} frames between lost frames (UMI's 24 at 60 fps)",
}


@dataclass(frozen=True)
class Piece:
    """A run of consecutive tracked frames ``[start, stop)`` (indices into times.txt) kept as its own episode.
    ``index``: 1-based among the kept pieces; ``t_first_ns``/``t_last_ns``: its first and last frame times."""

    index: int
    start: int
    stop: int
    t_first_ns: int
    t_last_ns: int

    @property
    def n_frames(self) -> int:
        return self.stop - self.start

    def to_json(self) -> dict:
        return {"index": self.index, "frames": [self.start, self.stop], "n_frames": self.n_frames,
                "t_first_ns": self.t_first_ns, "t_last_ns": self.t_last_ns,
                "seconds": (self.t_last_ns - self.t_first_ns) / 1e9}


@dataclass
class RulesResult:
    """The rules applied to one demo. ``ok``: at least one piece is kept (``pieces``) and no drop rule hit.
    ``reasons``: drop codes (:data:`REASONS`). ``short``: the tracked runs dropped for being shorter than
    :data:`MIN_PIECE_FRAMES` (``[start, stop)``)."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    n_frames: int = 0
    n_fed: int = 0
    n_tracked: int = 0
    n_lost: int = 0
    first_tracked_s: float | None = None
    stopped: bool = False
    pieces: list[Piece] = field(default_factory=list)
    short: list[tuple[int, int]] = field(default_factory=list)
    lost_runs: list[tuple[int, int]] = field(default_factory=list)

    @property
    def message(self) -> str:
        head = (f"{self.n_tracked}/{self.n_frames} frames tracked, {self.n_lost} lost (limit {MAX_LOST_IN_DEMO}), "
                f"first tracked at " + ("never" if self.first_tracked_s is None else f"{self.first_tracked_s:.2f} s")
                + f" (limit {MAX_RELOCALIZE_S} s)")
        if self.stopped:
            head = f"stopped after {self.n_fed} of {self.n_frames} frames ({MAX_LOST_FRAMES} not tracked); " + head
        if self.ok:
            sizes = ", ".join(str(p.n_frames) for p in self.pieces)
            head += f"; {len(self.pieces)} piece(s) of {sizes} frames"
            if self.short:
                head += f" ({len(self.short)} shorter than {MIN_PIECE_FRAMES} frames dropped)"
        return head

    def to_json(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "message": self.message, "n_frames": self.n_frames,
                "n_fed": self.n_fed, "n_tracked": self.n_tracked, "n_lost": self.n_lost,
                "first_tracked_s": self.first_tracked_s, "stopped": self.stopped,
                "pieces": [p.to_json() for p in self.pieces], "short_runs": [list(x) for x in self.short],
                "lost_runs": [list(x) for x in self.lost_runs], "limits": dict(LIMITS)}


def bool_segments(seq) -> list[tuple[int, int, bool]]:
    """``(start, stop, value)`` of every run of equal values, as UMI's ``get_bool_segments`` (``06:41-54``)."""
    b = np.asarray(seq, dtype=bool).reshape(-1)
    if len(b) == 0:
        return []
    ends = (np.flatnonzero(np.diff(b.astype(np.int8))) + 1).tolist()
    bounds = [0] + ends + [len(b)]
    return [(bounds[i], bounds[i + 1], bool(b[bounds[i]])) for i in range(len(bounds) - 1)]


def apply_rules(tracked, times_ns, *, stopped: bool, n_fed: int | None = None) -> RulesResult:
    """UMI's rules (module docstring) on one localization run.

    ``tracked``: per fed frame, visually tracked (``ForkTrajectory.tracked``; may be shorter than ``times_ns`` when
    the run stopped); ``times_ns``: every frame of times.txt; ``stopped``: the run stopped at ``--max_lost_frames``
    (the summary's ``stop_reason == "max_lost_frames"``); ``n_fed``: frames fed (default ``len(tracked)``)."""
    t = np.asarray(times_ns, dtype=np.int64).reshape(-1)
    n = int(len(t))
    tr = np.zeros(n, bool)
    given = np.asarray(tracked, bool).reshape(-1)[:n]
    tr[:len(given)] = given
    res = RulesResult(ok=False, n_frames=n, n_fed=int(len(given) if n_fed is None else n_fed),
                      n_tracked=int(tr.sum()), n_lost=int(n - tr.sum()), stopped=bool(stopped))
    if n == 0:
        res.reasons = ["too_few_tracked"]
        return res
    first = np.flatnonzero(tr)
    res.first_tracked_s = None if len(first) == 0 else (int(t[first[0]]) - int(t[0])) / 1e9
    segs = bool_segments(tr)
    res.lost_runs = [(a, b) for a, b, v in segs if not v]
    if stopped:                                       # 03 + gopro_slam.cc:255-258, 06:592-596
        res.reasons = ["stopped_lost"]
        return res
    if res.n_lost > MAX_LOST_IN_DEMO:                 # 06:604-608
        res.reasons.append("too_many_lost")
    if res.n_tracked < MIN_TRACKED_FRAMES:            # 06:610-614
        res.reasons.append("too_few_tracked")
    if res.first_tracked_s is None or res.first_tracked_s > MAX_RELOCALIZE_S:
        res.reasons.append("late_relocalization")
    # 06:719-726: tracked runs shorter than min_episode_length become invalid; 06:729-763: each remaining run is one
    # episode
    k = 0
    for a, b, v in segs:
        if not v:
            continue
        if b - a < MIN_PIECE_FRAMES:
            res.short.append((a, b))
            continue
        k += 1
        res.pieces.append(Piece(k, a, b, int(t[a]), int(t[b - 1])))
    if not res.pieces:
        res.reasons.append("no_piece")
    if res.reasons:
        res.pieces = []
        return res
    res.ok = True
    return res
