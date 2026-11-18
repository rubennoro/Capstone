"""Checks M1-M5 of a session map (docs/DESIGN.md "Checks -> map"), each with its value, its limit and, when it fails,
what to change in the recording. Pure functions of the SLAM runs' outputs, so every boundary is testable without the
native binary.

Only M1 and M5's sightings **block** the map (:data:`BLOCKING`); M2, M3, M4 and M5's spread and tilt limits are
**warnings**: printed with their hints and stored in the report, but the task folder is still created. The test demos
(DIRECTIONS.md section 3) then show whether the map works, as in UMI, which judges no map at all.

==  ==========================================================================================  =====================
#   passes when                                                                                 computed from
==  ==========================================================================================  =====================
M1  the mapping run finished and ends with exactly one map that has keyframes (and saved it)    mapping run: exit,
                                                                                                UMIPROC_SUMMARY maps
M2  that map's IMU is initialised and both inertial refinements (VIBA 1 and VIBA 2) finished    MapSummary flags
M3  the mapping clip re-localized against the saved map: first visually tracked frame (state    re-localized CSV
    OK, not lost) within 0.5 s of the first frame, and >= 95 % of ALL frames of the clip
    tracked (frames never fed, after a stop at --max_lost_frames, count as lost)
M4  the copied gravity check (|mean specific force| within 0.5 m/s^2 of g after the gravity     re-localized CSV +
    alignment) and metric-scale check (|s - 1| <= 0.1, observable) pass on the re-localized     the recording's IMU
    trajectory (visually tracked frames, as T_W_imu); the SLAM orientation agrees with
    gravity throughout (:func:`orientation_consistency`: every 2.5 s window's mean specific
    force within 3 deg of the clip's, >= 4 windows); the camera speed between consecutive
    tracked frames is <= 3 m/s
M5  table tag: >= 20 sightings pass UMI's filters (keyframes, 0.3-2 m, centre within 0.6 H/2);  re-localized CSV +
    the kept positions' per-axis std each <= 1 cm; the kept sightings' median normal within 5   tag detections
    deg of gravity's up
==  ==========================================================================================  =====================

M4's orientation sub-check: the copied gravity check runs after the copied gravity alignment, which rotates the world
onto the clip's mean specific force, so its angle is ~0 by construction and only |g| informs. The sub-check compares
the SLAM orientation with gravity *over time* instead: the tracked span is cut into windows of 2.5 s, each window's
mean of ``R_W_imu(t) f_imu(t)`` (IMU samples between tracked frames at most 0.2 s apart) is compared with the mean
over the whole span, and the largest angle must be <= 3 deg. A camera-IMU rotation error or a lens error makes the
estimated gravity turn with the rig, and a map that bends makes it drift, so different windows disagree; a consistent
trajectory keeps every window within the hand's mean acceleration over 2.5 s (a velocity change of 0.6 m/s is 0.24
m/s^2, 1.4 deg) plus an accelerometer bias seen from different orientations (20 mg, the LSM6DSOX's typical zero-g
offset, is 1.2 deg). 3 deg is the copied gravity check's own angle limit; a 5 deg camera-IMU rotation error shows as
up to 5 deg once the rig turns.

M5's spread: UMI prints the per-axis standard deviation of the kept positions (calibrate_slam_tag.py:97, 102-103); each
axis' std (ddof 0) must be <= 1 cm, and the 3-D value ``sqrt(sx^2 + sy^2 + sz^2)`` is reported too. The normal: the
median (Weiszfeld geometric median, renormalised) of the kept sightings' z axes (out of the tag) against the world's up,
which is the mean specific force of the re-localized trajectory in the SLAM world (the copied ``gravity_align``
estimate; the SLAM world is the first keyframe's camera frame, not gravity-aligned). The normal of the one sighting
UMI keeps as ``tx_slam_tag`` is reported too: a single sighting's rotation is noisier than its position.

M1/M2 hint after many map resets right after initialisation (:func:`init_resets`): when the mapping run reset its map
>= 5 times and most new maps were lost within 15 tracked frames (0.5 s; "Fail to track local map!": too few matches to
its own fresh map), the cause is most likely the camera model or the camera-IMU geometry, not the recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from umiproc.slam.log import STATE_OK

__all__ = ["BLOCKING", "CheckResult", "M3_MAX_RELOCALIZE_S", "M3_MIN_TRACKED_FRACTION", "M4_MAX_SPEED_MPS", "M4_WINDOW_S",
           "M4_MAX_WINDOW_DEG", "M4_MIN_WINDOWS", "M5_MIN_SIGHTINGS", "M5_MAX_SPREAD_M", "M5_MAX_NORMAL_DEG",
           "RELOCALIZE_MAX_LOST_FRAMES", "PACING_MIN_FRACTION", "EARLY_RESET_MAX_FRAMES", "EARLY_RESETS_MIN", "LIMITS",
           "check_m1", "check_m2", "check_m3", "check_m4", "check_m5", "not_evaluated", "max_speed",
           "lost_stretches", "the_map", "pacing_warnings", "imu_trajectory", "up_direction",
           "orientation_consistency", "init_resets", "tracked_frames_before_restart", "FIRST_MAP_MIN_TRACKED",
           "MapPiece", "map_pieces", "first_map_piece"]

M3_MAX_RELOCALIZE_S = 0.5
M3_MIN_TRACKED_FRACTION = 0.95
M4_MAX_SPEED_MPS = 3.0              # the copied validate's MAX_SPEED_MPS (iPhUMI align_validate)
#: M4's orientation sub-check (module docstring): windows of this length over the tracked span ...
M4_WINDOW_S = 2.5
#: ... each window's mean specific force within this angle of the span's (the copied gravity check's angle limit) ...
M4_MAX_WINDOW_DEG = 3.0
#: ... with at least this many windows (10 s tracked), else "not enough tracked span"
M4_MIN_WINDOWS = 4
#: a window counts when its usable IMU samples cover at least this fraction of it
M4_WINDOW_MIN_COVERAGE = 0.5
#: IMU samples are used only between tracked frames at most this far apart (the copied scale_check's max gap)
M4_MAX_POSE_GAP_S = 0.2
M5_MIN_SIGHTINGS = 20
M5_MAX_SPREAD_M = 0.01              # per axis (std, ddof 0)
M5_MAX_NORMAL_DEG = 5.0
#: UMI's 03_batch_slam.py:44 stops localization after 60 lost frames at 60 fps (1 s): 30 at our 30 fps.
RELOCALIZE_MAX_LOST_FRAMES = 30
#: Below this fraction of the recording's frame rate, SLAM ran slower than real time (a diagnostic, not a check).
PACING_MIN_FRACTION = 0.9
#: :func:`init_resets`: a new map lost within this many visually tracked frames (0.5 s at 30 fps) was lost "right
#: after initialisation" ...
EARLY_RESET_MAX_FRAMES = 15
#: ... and from this many map resets on, with most of them that early, M1/M2 point at the camera model.
EARLY_RESETS_MIN = 5
LIMITS = {"M1": {"maps_with_keyframes": 1}, "M2": {"imu_initialized": True, "viba1": True, "viba2": True},
          "M3": {"first_tracked_s_max": M3_MAX_RELOCALIZE_S, "tracked_fraction_min": M3_MIN_TRACKED_FRACTION},
          "M4": {"gravity": "copied gravity_check (| |g| - 9.81 | < 0.5 m/s^2; its angle is ~0 after the alignment)",
                 "orientation": {"window_s": M4_WINDOW_S, "max_window_to_mean_deg": M4_MAX_WINDOW_DEG,
                                 "windows_min": M4_MIN_WINDOWS},
                 "scale": "copied scale_check status ok (|s - 1| <= 0.1, observable)",
                 "max_speed_mps": M4_MAX_SPEED_MPS},
          "M5": {"sightings_min": M5_MIN_SIGHTINGS, "spread_per_axis_m_max": M5_MAX_SPREAD_M,
                 "median_normal_to_up_deg_max": M5_MAX_NORMAL_DEG, "distance_m": [0.3, 2.0],
                 "centre_fraction_max": 0.6}}

TITLES = {"M1": "one map with keyframes", "M2": "IMU fully initialised (VIBA 1 and 2)",
          "M3": "the mapping clip re-localizes against the map", "M4": "gravity, metric scale and speed",
          "M5": "table tag 13 in the map"}

_START_HINT = ("start the clip held still 30-50 cm above tag 13 lying flat on the table for about 3 s, then small "
               "slow up/down and side-to-side motions for about 5 s")
_SCAN_HINT = ("scan the workspace slowly for at least 30 s without stopping: real translation (not only "
              "turning on the spot), no fast moves, never a blank view (a plain wall, the ceiling, the lens covered)")


#: Whether a check outside its limit stops the map (:attr:`CheckResult.blocking`; branch rnoro/data-work-auto,
#: 2026-11-17). UMI judges no map at all; the test demos (DIRECTIONS.md section 3) are the gate. Only what keeps the
#: data correct still blocks: M1 (one map: demos and the table tag in the same coordinates) and M5's sightings (a
#: table pose to exist; :func:`check_m5` sets it per result: its spread and tilt limits only warn). M2-M4 warn.
BLOCKING = {"M1": True, "M2": False, "M3": False, "M4": False, "M5": True}


@dataclass
class CheckResult:
    """One check: ``passed`` (None = could not be evaluated, which counts as not passed), its values and limits, a
    one-line ``message`` (value against limit) and, when it did not pass, ``hints`` (what to change in the
    recording). ``blocking``: whether not passing stops the map (default :data:`BLOCKING`); a non-blocking check that
    did not pass is a **warning** (:attr:`warned`)."""

    id: str
    passed: bool | None
    values: dict = field(default_factory=dict)
    message: str = ""
    hints: list[str] = field(default_factory=list)
    blocking: bool | None = None

    def __post_init__(self):
        if self.blocking is None:
            self.blocking = BLOCKING[self.id]

    @property
    def title(self) -> str:
        return TITLES[self.id]

    @property
    def ok(self) -> bool:
        return self.passed is True

    @property
    def failed(self) -> bool:
        """Did not pass and blocks the map."""
        return not self.ok and self.blocking

    @property
    def warned(self) -> bool:
        """Did not pass, but only warns."""
        return not self.ok and not self.blocking

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "passed": self.passed, "blocking": bool(self.blocking),
                "values": _jsonable(self.values), "limits": LIMITS[self.id], "message": self.message,
                "hints": list(self.hints)}


def _jsonable(x):
    from umiproc.dp.files import jsonable

    return jsonable(x)


def not_evaluated(check_id: str, why: str, hints: list[str] | None = None) -> CheckResult:
    return CheckResult(check_id, None, {"evaluated": False}, f"not evaluated: {why}", list(hints or []))


# --------------------------------------------------------------------------------------------- M1, M2

def _did_not_finish(returncode, timed_out: bool, summary) -> str | None:
    if timed_out:
        return "the run timed out (killed)"
    if summary is None:
        return f"the run died before its summary line (exit {returncode})"
    if returncode != 0 or summary.exit_code != 0:
        return f"the run exited with {returncode}" + (f": {summary.error}" if summary.error else "")
    return None


def _signal_name(returncode) -> str | None:
    """``SIGSEGV`` for exit -11 (a process killed by a signal), else None."""
    import signal

    if isinstance(returncode, int) and returncode < 0:
        try:
            return signal.Signals(-returncode).name
        except ValueError:
            return f"signal {-returncode}"
    return None


_LOG_HINT = "mapping/umiproc_slam.log next to report.json"


def tracked_frames_before_restart(states) -> list[int]:
    """From the mapping run's per-frame tracking states (its trajectory CSV): for every map that was dropped and
    started again (a stretch of frames with a map, followed by NOT_INITIALIZED / NO_IMAGES_YET again), how many of its
    frames were visually tracked (state OK). The last stretch (the map that survived) is not included."""
    st = np.asarray(states, dtype=np.int64)
    out, cur = [], None
    for v in st:
        if v <= 1:                                         # SYSTEM_NOT_READY, NO_IMAGES_YET, NOT_INITIALIZED
            if cur is not None:
                out.append(cur)
                cur = None
        else:
            cur = (0 if cur is None else cur) + int(v == 2)
    return out


#: ``map --first-map``: a piece needs more visually tracked frames than a map dropped right after initialising
FIRST_MAP_MIN_TRACKED = EARLY_RESET_MAX_FRAMES + 1


@dataclass(frozen=True)
class MapPiece:
    """One stretch of fed frames with a map (state > NOT_INITIALIZED), between two map (re)starts."""

    start: int              # first fed frame of the stretch
    end: int                # last fed frame of the stretch
    first_tracked: int      # first / last fed frame with state OK (-1: none)
    last_tracked: int
    tracked: int            # frames with state OK

    def to_json(self, t_s=None) -> dict:
        out = {"start": self.start, "end": self.end, "first_tracked": self.first_tracked,
               "last_tracked": self.last_tracked, "tracked": self.tracked}
        if t_s is not None and self.tracked:
            out.update(first_tracked_s=round(float(t_s[self.first_tracked]), 3),
                       last_tracked_s=round(float(t_s[self.last_tracked]), 3))
        return out


def map_pieces(states) -> list[MapPiece]:
    """The mapping run's stretches of frames with a map, in order (the same split as
    :func:`tracked_frames_before_restart`, the last stretch included): a new stretch starts when the state falls back
    to NOT_INITIALIZED / NO_IMAGES_YET / SYSTEM_NOT_READY, i.e. SLAM dropped its map and started a new one."""
    st = np.asarray(states, dtype=np.int64)
    out: list[MapPiece] = []
    start = None
    for k in range(len(st) + 1):
        inside = k < len(st) and st[k] > 1
        if inside and start is None:
            start = k
        elif not inside and start is not None:
            ok = np.flatnonzero(st[start:k] == STATE_OK) + start
            out.append(MapPiece(start, k - 1, int(ok[0]) if len(ok) else -1, int(ok[-1]) if len(ok) else -1,
                                int(len(ok))))
            start = None
    return out


def first_map_piece(states, min_tracked: int = FIRST_MAP_MIN_TRACKED) -> MapPiece | None:
    """The first piece that tracked at least ``min_tracked`` frames (maps dropped right after initialising are
    skipped), or None."""
    return next((p for p in map_pieces(states) if p.tracked >= min_tracked), None)


def init_resets(log, states=None) -> dict:
    """Whether the mapping run kept losing its new map right after initialising it (module docstring).

    ``log``: the mapping run's :class:`~umiproc.slam.log.SlamLog`; ``states``: its trajectory CSV's per-frame states
    (None when it could not be read). With ``states``, "right after" means within :data:`EARLY_RESET_MAX_FRAMES`
    visually tracked frames (:func:`tracked_frames_before_restart`); without, a reset whose map printed nothing but
    "Fail to track local map!" between "New Map created" and the reset. ``suspect``: >= :data:`EARLY_RESETS_MIN`
    resets and at least half of the dropped maps that early."""
    events = list(log.events) if log is not None else []
    resets = sum(1 for e in events if e.kind == "map_reset")
    after_init = 0                                         # the log pattern
    since_init: list[str] | None = None
    for e in events:
        if e.kind == "map_initialized":
            since_init = []
        elif e.kind == "map_reset":
            if since_init is not None and since_init and set(since_init) == {"local_map_failed"}:
                after_init += 1
            since_init = None
        elif since_init is not None:
            since_init.append(e.kind)
    out = {"resets": resets, "local_map_failed_right_after_init": after_init,
           "early_max_frames": EARLY_RESET_MAX_FRAMES, "resets_min": EARLY_RESETS_MIN}
    if states is not None:
        tracked = tracked_frames_before_restart(states)
        early = sum(1 for n in tracked if n <= EARLY_RESET_MAX_FRAMES)
        out.update(source="mapping trajectory", tracked_frames_before_restart=tracked, early=early,
                   dropped_maps=len(tracked))
    else:
        early, out["source"], out["dropped_maps"] = after_init, "mapping log", resets
        out["early"] = early
    out["suspect"] = bool(resets >= EARLY_RESETS_MIN and out["dropped_maps"] > 0
                          and 2 * out["early"] >= out["dropped_maps"])
    return out


def _camera_model_hint(early: dict) -> str:
    frames = (f"within {EARLY_RESET_MAX_FRAMES} tracked frames (0.5 s)" if early.get("source") == "mapping trajectory"
              else "right after \"New Map created\"")
    return (f"SLAM reset its map {early['resets']} times, and {early['early']} of the {early['dropped_maps']} new maps "
            f"were lost {frames} (\"Fail to track local map!\": too few matches to its own fresh map). That is most "
            "likely the camera model, not the recording: calibrate the lens (DIRECTIONS.md section 1, then "
            "`umiproc lens-calibrate ... --apply`) and check the camera-IMU geometry (cameras.ultrawide.T_imu_cam in "
            "the rig file); a longer or slower clip will not fix it. Logs: " + _LOG_HINT)


def check_m1(returncode, timed_out: bool, summary, log, map_saved: bool, early: dict | None = None) -> CheckResult:
    """M1 on the mapping run (``log``: its :class:`~umiproc.slam.log.SlamLog`; ``early``: :func:`init_resets`)."""
    resets = [e.detail.get("reason") for e in log.resets] if log is not None else []
    inits = [e.detail.get("method") for e in log.of_kind("map_initialized")] if log is not None else []
    values = {"map_resets": len(resets), "reset_reasons": resets, "initializations": inits, "init_resets": early}
    bad = _did_not_finish(returncode, timed_out, summary)
    if bad:
        values["finished"] = False
        sig = _signal_name(returncode)
        if sig is not None:
            hint = (f"the SLAM program crashed ({sig}, exit {returncode}): this is not a recording problem: read the "
                    f"end of the mapping log ({_LOG_HINT}) and report it")
        else:
            hint = f"this is not a recording problem: read the mapping log ({_LOG_HINT}) and report it"
        return CheckResult("M1", False, values, f"the mapping run did not finish: {bad}", [hint])
    maps = summary.maps or ()
    with_kf = summary.maps_with_keyframes
    values.update(finished=True, maps=[{"id": m.id, "keyframes": m.keyframes, "map_points": m.map_points}
                                       for m in maps], maps_with_keyframes=len(with_kf), map_saved=bool(map_saved))
    if len(with_kf) == 1 and map_saved:
        m = with_kf[0]
        return CheckResult("M1", True, values, f"1 map with keyframes ({m.keyframes} keyframes, {m.map_points} "
                                               f"points); {len(resets)} map reset(s) on the way")
    hints = []
    suspect = bool(early and early.get("suspect"))
    if not with_kf:
        msg = "no map with keyframes at the end of the mapping run"
        if not inits:
            hints.append("SLAM never initialised: " + _START_HINT + "; make sure tag 13 (DICT_4X4_50, 16 cm) is "
                         "fully in view and sharp at the start, the light is good and the scene has texture "
                         "(not a plain table)")
        elif suspect:
            hints.append(_camera_model_hint(early))
        else:
            hints.append(f"SLAM initialised {len(inits)} time(s) but lost every map ({len(resets)} reset(s): "
                         f"{_reasons(resets)}): a new map is dropped when tracking is lost before the IMU is fully "
                         "initialised (about 15 s), so the first 15 s must stay tracked: " + _START_HINT + "; then "
                         + _SCAN_HINT)
    elif len(with_kf) > 1:
        msg = (f"{len(with_kf)} maps with keyframes (SLAM lost tracking and started a new map "
               f"{len(with_kf) - 1} time(s)); one map is needed")
        hints.append(_camera_model_hint(early) if suspect else
                     "tracking was lost during the scan and a second map was started: " + _SCAN_HINT)
    else:
        msg = "the map file was not saved"
        hints.append(f"this is not a recording problem: read the mapping log ({_LOG_HINT})")
    return CheckResult("M1", False, values, msg, hints)


def _reasons(resets) -> str:
    names = {"lost_before_viba2": "lost before the IMU was fully initialised", "bad_imu": "bad IMU initialisation",
             "timestamp_jump": "a time jump", "wrong_initialization": "wrong initialisation",
             "lost_before_imu_init": "lost before the IMU initialisation", None: "reason not printed"}
    out = {}
    for r in resets:
        out[names.get(r, r)] = out.get(names.get(r, r), 0) + 1
    return ", ".join(f"{k} x{v}" if v > 1 else k for k, v in out.items()) or "none"


def the_map(summary):
    """The map M2 judges: the only map with keyframes, else the one with the most."""
    maps = summary.maps_with_keyframes if summary is not None else ()
    return max(maps, key=lambda m: m.keyframes) if maps else None


def check_m2(summary, early: dict | None = None) -> CheckResult:
    """M2 on the mapping run's summary (``early``: :func:`init_resets`, which changes the hint)."""
    m = the_map(summary)
    if m is None:
        return not_evaluated("M2", "there is no map with keyframes (M1)")
    values = {"map_id": m.id, "keyframes": m.keyframes, "imu_initialized": m.imu_initialized,
              "viba1": m.inertial_ba1, "viba2": m.inertial_ba2}
    if m.imu_initialized and m.inertial_ba1 and m.inertial_ba2:
        return CheckResult("M2", True, values, "IMU initialised, VIBA 1 and VIBA 2 finished")
    missing = [n for n, ok in (("IMU initialisation", m.imu_initialized), ("VIBA 1", m.inertial_ba1),
                               ("VIBA 2", m.inertial_ba2)) if not ok]
    if early and early.get("suspect"):
        hints = [_camera_model_hint(early)]
    else:
        hints = ["ORB-SLAM3 refines the IMU twice while it keeps tracking: VIBA 1 about 5 s and VIBA 2 about 15 s "
                 "after the IMU initialisation. Record longer (at least 30 s of continuous tracking): " + _SCAN_HINT,
                 "the IMU needs to see real accelerations and rotations in every direction: mix gentle rotation "
                 "with translation (up/down, side to side, forward/back), never only turning on the spot"]
    return CheckResult("M2", False, values, "not finished: " + ", ".join(missing), hints)


# --------------------------------------------------------------------------------------------- M3

def lost_stretches(tracked: np.ndarray, t_ns: np.ndarray, t0_ns: int, top: int = 3) -> list[dict]:
    """The ``top`` longest runs of not-tracked frames: ``{start_s, frames, seconds}`` (seconds since ``t0_ns``)."""
    tr = np.asarray(tracked, bool)
    if not len(tr):
        return []
    d = np.diff(np.concatenate([[1], tr.astype(np.int8), [1]]))
    starts, ends = np.flatnonzero(d == -1), np.flatnonzero(d == 1)
    out = []
    for a, b in zip(starts, ends):
        t_a, t_b = int(t_ns[a]), int(t_ns[b - 1])
        frame_s = float(np.median(np.diff(t_ns))) * 1e-9 if len(t_ns) > 1 else 0.0
        out.append({"start_s": (t_a - t0_ns) / 1e9, "frames": int(b - a), "seconds": (t_b - t_a) / 1e9 + frame_s})
    out.sort(key=lambda r: -r["frames"])
    return out[:top]


def check_m3(returncode, timed_out: bool, summary, log, traj, times_ns: np.ndarray) -> CheckResult:
    """M3 on the re-localization run. ``traj``: its ForkTrajectory (None when it could not be read);
    ``times_ns``: every frame of the clip (times.txt)."""
    t = np.asarray(times_ns, dtype=np.int64)
    n_total = int(len(t))
    values = {"frames": n_total}
    if timed_out or summary is None or traj is None:
        why = "timed out" if timed_out else ("died before its summary line" if summary is None
                                              else "its trajectory could not be read")
        values["finished"] = False
        return CheckResult("M3", False, values, f"the re-localization run {why}",
                           ["this is not a recording problem: read the re-localization log (umiproc_slam.log)"])
    tracked = np.zeros(n_total, bool)
    tracked[:traj.n_frames] = np.asarray(traj.tracked, bool)
    n_tr = int(tracked.sum())
    frac = n_tr / n_total if n_total else 0.0
    first = traj.first_tracked_index
    first_s = None if first is None else (int(t[first]) - int(t[0])) / 1e9
    values.update(finished=True, exit_code=returncode, stopped_early=bool(summary.stopped_early),
                  stop_reason=summary.stop_reason, frames_fed=summary.frames_fed, frames_tracked=n_tr,
                  tracked_fraction=frac, first_tracked_s=first_s,
                  relocalizations=len(log.of_kind("relocalized")) if log is not None else None,
                  relocalization_times_s=log.relocalization_times_s if log is not None else None,
                  longest_lost=lost_stretches(tracked, t, int(t[0])))
    ok_first = first_s is not None and first_s <= M3_MAX_RELOCALIZE_S
    ok_frac = frac >= M3_MIN_TRACKED_FRACTION
    msg = (f"first tracked frame at {'never' if first_s is None else f'{first_s:.2f} s'} (limit "
           f"{M3_MAX_RELOCALIZE_S} s); {n_tr}/{n_total} frames tracked = {100 * frac:.1f} % (limit "
           f">= {100 * M3_MIN_TRACKED_FRACTION:.0f} %)")
    if summary.stopped_early:
        msg += f"; stopped after {summary.frames_fed} frames ({RELOCALIZE_MAX_LOST_FRAMES} lost in a row)"
    if ok_first and ok_frac:
        return CheckResult("M3", True, values, msg)
    hints = []
    if not ok_first:
        hints.append("the clip's first half second must show what the map knows best: start exactly as the mapping "
                     "start (still, 30-50 cm above tag 13), with good light")
    if not ok_frac:
        worst = values["longest_lost"][0] if values["longest_lost"] else None
        where = (f" (the longest loss: {worst['frames']} frames from {worst['start_s']:.1f} s)" if worst else "")
        hints.append("tracking was lost against the map" + where + ": " + _SCAN_HINT + "; nothing in the scene "
                     "may move during the clip, and the scan should revisit the start area")
    return CheckResult("M3", False, values, msg, hints)


# --------------------------------------------------------------------------------------------- M4

def imu_trajectory(traj, T_imu_cam: np.ndarray):
    """The visually tracked rows of a ForkTrajectory as a copied :class:`~umiproc.dp.geometry.trajectory.Trajectory`
    of the IMU (``T_W_imu = T_W_cam @ inv(T_imu_cam)``), world = the SLAM world (the first keyframe's camera)."""
    from umiproc.dp.geometry.trajectory import Trajectory

    sel = np.asarray(traj.tracked, bool)
    T_cam_imu = np.linalg.inv(np.asarray(T_imu_cam, dtype=np.float64))
    T = np.einsum("nij,jk->nik", traj.T_W_cam[sel], T_cam_imu)
    return Trajectory.from_T(traj.t_ns[sel], T, frame="imu", world="slam_first_keyframe")


def max_speed(traj) -> tuple[float | None, float | None]:
    """Largest camera speed (m/s) between consecutive frames that are both visually tracked, and its time (s since
    the first frame); (None, None) without such a pair."""
    tr = np.asarray(traj.tracked, bool)
    pair = tr[:-1] & tr[1:]
    if not pair.any():
        return None, None
    p = traj.T_W_cam[:, :3, 3]
    dt = np.diff(traj.t_ns).astype(np.float64) * 1e-9
    v = np.full(len(dt), -np.inf)
    v[pair] = np.linalg.norm(np.diff(p, axis=0)[pair], axis=1) / dt[pair]
    k = int(np.argmax(v))
    return float(v[k]), (int(traj.t_ns[k]) - int(traj.t_ns[0])) / 1e9


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def orientation_consistency(traj, imu_t_ns, acc_imu, *, t_ref_ns: int | None = None,
                            window_s: float = M4_WINDOW_S) -> dict:
    """M4's orientation sub-check (module docstring): does the SLAM orientation agree with gravity over the whole
    tracked span?

    ``traj``: the visually tracked poses as a copied ``Trajectory`` of the IMU (:func:`imu_trajectory`, any rigid
    world); ``imu_t_ns``/``acc_imu``: the recording's IMU samples (IMU time base, specific force in the IMU frame,
    m/s^2). The samples between tracked frames at most :data:`M4_MAX_POSE_GAP_S` apart are rotated into the world
    (``R_W_imu(t) f``); the span from the first to the last tracked frame is cut into whole windows of ``window_s``;
    a window counts when its samples cover >= :data:`M4_WINDOW_MIN_COVERAGE` of it. Returns ``{ok, windows,
    windows_skipped, max_deg, median_deg, max_at_s (start of the worst window, s since t_ref_ns or the first tracked
    frame), window_deg (each counted window's angle), up_world (the span's mean direction), reason}``."""
    if getattr(traj, "frame", "imu") != "imu":
        raise ValueError(f"orientation_consistency needs an imu-frame trajectory, got {traj.frame!r}")
    out = {"ok": False, "window_s": float(window_s), "windows": 0, "windows_skipped": 0, "max_deg": None,
           "median_deg": None, "max_at_s": None, "window_deg": [], "up_world": None, "reason": None,
           "limit_deg": M4_MAX_WINDOW_DEG, "windows_min": M4_MIN_WINDOWS}
    t = np.asarray(imu_t_ns, dtype=np.int64)
    f = np.asarray(acc_imu, dtype=np.float64).reshape(-1, 3)
    if len(traj) < 2 or len(t) < 2:
        out["reason"] = "too few tracked poses or IMU samples"
        return out
    T, valid = traj.sample(t, max_gap_ns=int(round(M4_MAX_POSE_GAP_S * 1e9)))
    if not valid.any():
        out["reason"] = "no IMU sample inside the tracked span"
        return out
    f_W = np.einsum("nij,nj->ni", T[:, :3, :3], f)
    up = f_W[valid].mean(axis=0)
    out["up_world"] = (up / max(float(np.linalg.norm(up)), 1e-12)).tolist()
    rate_hz = 1e9 / float(np.median(np.diff(t)))
    w_ns = int(round(window_s * 1e9))
    t0, t1 = int(traj.t_ns[0]), int(traj.t_ns[-1])
    ref = t0 if t_ref_ns is None else int(t_ref_ns)
    need = M4_WINDOW_MIN_COVERAGE * window_s * rate_hz
    angles, starts = [], []
    for k in range((t1 - t0) // w_ns):
        a = t0 + k * w_ns
        sel = valid & (t >= a) & (t < a + w_ns)
        if sel.sum() < need:
            out["windows_skipped"] += 1
            continue
        angles.append(_angle_deg(f_W[sel].mean(axis=0), up))
        starts.append((a - ref) / 1e9)
    out["windows"] = len(angles)
    out["window_deg"] = [round(x, 3) for x in angles]
    if angles:
        k = int(np.argmax(angles))
        out.update(max_deg=float(angles[k]), median_deg=float(np.median(angles)), max_at_s=float(starts[k]))
    if len(angles) < M4_MIN_WINDOWS:
        out["reason"] = (f"not enough tracked span: {len(angles)} window(s) of {window_s:g} s (need "
                         f"{M4_MIN_WINDOWS}, about {M4_MIN_WINDOWS * window_s:g} s tracked)")
        return out
    out["ok"] = bool(out["max_deg"] <= M4_MAX_WINDOW_DEG)
    return out


def check_m4(slam_result, error: str | None, speed: tuple[float | None, float | None],
             orient: dict | None) -> CheckResult:
    """M4 from the copied ``finalize_trajectory`` result (its ``diagnostics`` gravity_check / scale_check),
    :func:`orientation_consistency` (``orient``; None when it could not be computed) and :func:`max_speed`.
    ``error``: why there is no result (too few tracked poses)."""
    if slam_result is None:
        return CheckResult("M4", False, {"evaluated": False, "orientation": orient},
                           f"no trajectory to check: {error}", ["see M3: the re-localized clip must be tracked"])
    d = slam_result.diagnostics
    g, s = d.get("gravity_check", {}), d.get("scale_check", {})
    o = orient or {"ok": False, "windows": 0, "reason": "not computed"}
    vmax, at_s = speed
    values = {"gravity_ok": bool(g.get("ok")), "g_norm": g.get("g_norm"), "angle_to_up_deg": g.get("angle_to_up_deg"),
              "gravity_alignment_deg": d.get("gravity_alignment", {}).get("angle_deg"),
              "orientation_ok": bool(o.get("ok")), "orientation_max_deg": o.get("max_deg"),
              "orientation_median_deg": o.get("median_deg"), "orientation_windows": o.get("windows"),
              "orientation": o,
              "scale_status": s.get("status"), "scale": s.get("scale"), "scale_std_err": s.get("std_err"),
              "scale_excitation_rms": s.get("excitation_rms"), "scale_problems": s.get("problems", []),
              "max_speed_mps": vmax, "max_speed_at_s": at_s}
    ok_g = bool(g.get("ok"))
    ok_o = bool(o.get("ok"))
    ok_s = s.get("status") == "ok"
    ok_v = vmax is not None and vmax <= M4_MAX_SPEED_MPS
    parts = [f"gravity {'ok' if ok_g else 'FAILED'} (|g| {_f(g.get('g_norm'), '.3f')} m/s^2, limit 9.807 +- 0.5)",
             f"orientation vs gravity {'ok' if ok_o else 'FAILED'} (max {_f(o.get('max_deg'), '.2f')} deg, median "
             f"{_f(o.get('median_deg'), '.2f')} deg over {o.get('windows', 0)} windows of {M4_WINDOW_S:g} s; limit "
             f"{M4_MAX_WINDOW_DEG:g} deg, >= {M4_MIN_WINDOWS} windows)",
             f"scale {s.get('status')} (s = {_f(s.get('scale'), '.3f')}, limit 1 +- 0.1)",
             f"max speed {_f(vmax, '.2f')} m/s (limit {M4_MAX_SPEED_MPS})"]
    if ok_g and ok_o and ok_s and ok_v:
        return CheckResult("M4", True, values, "; ".join(parts))
    hints = []
    if not ok_g:
        hints.append("the trajectory's orientations disagree with the IMU's gravity: move smoothly (no jerks or "
                     "bumps), and keep the clip tracked throughout (M3)")
    if not ok_o:
        if o.get("max_deg") is not None and (o.get("windows") or 0) >= M4_MIN_WINDOWS:
            hints.append(f"the SLAM orientation disagrees with gravity by up to {o['max_deg']:.1f} deg (the "
                         f"{M4_WINDOW_S:g} s window from {_f(o.get('max_at_s'), '.1f')} s; median "
                         f"{o['median_deg']:.1f} deg): gravity seen through SLAM turns with the rig or drifts. With a "
                         "smooth clip this is not the recording: check the camera-IMU rotation "
                         "(cameras.ultrawide.T_imu_cam in the rig file) and the lens model (DIRECTIONS.md section 1); "
                         "if M3 lost tracking, the map itself may bend")
        else:
            hints.append(f"the SLAM orientation could not be compared with gravity: {o.get('reason')}; the clip must "
                         "stay tracked (M3) for well over 10 s")
    if s.get("status") == "scale_unobservable":
        hints.append("the metric scale could not be measured: add more translation with changes of direction "
                     "(slow up/down and side-to-side strokes of 20-40 cm), not only rotation")
    elif not ok_s:
        hints.append(f"the map's metric scale is off by {abs((s.get('scale') or 1.0) - 1.0) * 100:.0f} %: give the "
                     "IMU clear accelerations early (the slow up/down and side-to-side motions right after the "
                     "still start), then keep tracking continuous")
    if not ok_v:
        hints.append(f"a camera speed of {_f(vmax, '.2f')} m/s at {_f(at_s, '.1f')} s: move slower (a fast swing, or "
                     "a SLAM jump right after a loss of tracking)")
    return CheckResult("M4", False, values, "; ".join(parts), hints)


def _f(v, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


# --------------------------------------------------------------------------------------------- M5

def up_direction(slam_result) -> np.ndarray | None:
    """World up in the SLAM world: the copied gravity alignment's mean specific force (normalised), or None."""
    if slam_result is None:
        return None
    m = slam_result.diagnostics.get("gravity_alignment", {}).get("mean_world_before")
    if m is None:
        return None
    m = np.asarray(m, dtype=np.float64)
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else None


def check_m5(stats, agg: dict | None, up: np.ndarray | None, frames_with_tag: int, frames_total: int) -> CheckResult:
    """M5 from :func:`umiproc.map.tag.collect_sightings` (``stats``), :func:`~umiproc.map.tag.aggregate_sightings`
    (``agg``, None without a sighting) and :func:`up_direction` (``up``). ``frames_with_tag``: frames of the clip in
    which the tag was detected at all (for the hints). The spread is judged per axis, the normal on the kept
    sightings' median normal (module docstring)."""
    n = len(stats.sightings) if stats is not None else 0
    values = {"frames_with_tag": frames_with_tag, "frames": frames_total, "sightings": n,
              "filters": stats.to_json() if stats is not None else None}
    hints = []
    ok_n = n >= M5_MIN_SIGHTINGS
    std = normal = None
    if agg is not None:
        std = np.asarray(agg["std_m"], float)
        values.update(std_m=std.tolist(), spread_max_axis_m=float(std.max()),
                      spread_3d_m=float(np.linalg.norm(std)), kept_after_outliers=int(np.sum(agg["kept"])),
                      outlier_threshold_m=agg["threshold_m"])
        if up is not None:
            normal = _angle_deg(np.asarray(agg["normal_median"], float), np.asarray(up, float))
            chosen = (np.asarray(agg["normal_chosen"], float) if agg.get("normal_chosen") is not None
                      else np.asarray(agg["T_world_tag"], float)[:3, 2])
            values.update(normal_to_up_deg=normal, chosen_normal_to_up_deg=_angle_deg(chosen, np.asarray(up, float)))
    ok_spread = std is not None and bool(np.all(std <= M5_MAX_SPREAD_M))
    ok_normal = normal is not None and normal <= M5_MAX_NORMAL_DEG
    spread_s = "n/a" if std is None else "/".join(f"{100 * v:.2f}" for v in std) + " cm"
    msg = (f"{n} sightings (limit >= {M5_MIN_SIGHTINGS}); spread x/y/z {spread_s} (limit {100 * M5_MAX_SPREAD_M:.0f} "
           f"cm each{'' if std is None else f'; 3-D {100 * float(np.linalg.norm(std)):.2f} cm'}); median normal "
           f"{'n/a' if normal is None else f'{normal:.1f} deg'} from up (limit {M5_MAX_NORMAL_DEG:.0f} deg"
           + ("" if values.get("chosen_normal_to_up_deg") is None else
              f"; the chosen sighting's {values['chosen_normal_to_up_deg']:.1f} deg") + ")")
    if ok_n and ok_spread and ok_normal:
        return CheckResult("M5", True, values, msg)
    # blocks only without a usable table pose (too few sightings); the spread and tilt limits only warn
    blocking = not ok_n or agg is None
    if not ok_n:
        st = stats
        if frames_with_tag == 0:
            hints.append("tag 13 was never detected: print it (DICT_4X4_50, id 13, black square 16 cm), lay it flat "
                         "on the table in the workspace, uncovered, and look at it at the start and during the scan")
        else:
            why = []
            if st is not None and st.keyframes_with_tag < M5_MIN_SIGHTINGS:
                why.append(f"only {st.keyframes_with_tag} keyframes see it (seen in {frames_with_tag} of "
                           f"{frames_total} frames): look at it longer, from several places")
            if st is not None and st.rejected_distance:
                why.append(f"{st.rejected_distance} were nearer than 0.3 m or farther than 2 m")
            if st is not None and st.rejected_centre:
                why.append(f"{st.rejected_centre} had the tag too far from the image centre (keep it in the middle "
                           "of the view)")
            hints.append("too few usable views of tag 13: " + "; ".join(why) if why else
                         "too few usable views of tag 13: keep it in the middle of the view, 30-50 cm away, for the "
                         "first seconds and again during the scan")
    if std is not None and not ok_spread:
        axis = "xyz"[int(np.argmax(std))]
        hints.append(f"the tag's position in the map varies by {100 * float(std.max()):.1f} cm (std along {axis}): "
                     "the tag must lie flat and never move; if it did not move, the map is distorted (see M3/M4: "
                     "slow, continuous tracking)")
    if agg is not None and not ok_normal:
        if up is None:
            hints.append("gravity's direction in the map is unknown (see M4)")
        else:
            hints.append(f"the tag is tilted {normal:.1f} deg from level (the median of its views): tape it flat on "
                         "a level table; if it is flat, the map's gravity is off (see M4)")
    return CheckResult("M5", False, values, msg, hints, blocking=blocking)


# --------------------------------------------------------------------------------------------- diagnostics

def pacing_warnings(name: str, summary) -> list[str]:
    """A plain warning when the run fed frames below :data:`PACING_MIN_FRACTION` of the recording's rate."""
    p = getattr(summary, "pacing", None)
    if p is None or p.recording_rate_hz <= 0:
        return []
    if p.feed_rate_hz < PACING_MIN_FRACTION * p.recording_rate_hz:
        return [f"{name}: SLAM ran slower than real time: fed {p.feed_rate_hz:.2f} of {p.recording_rate_hz:.2f} "
                f"frames/s ({p.frames_over_period} frames took longer than their frame period; tracking median "
                f"{p.track_ms_median:.0f} ms). ORB-SLAM3 then skips keyframes, so the map can be worse than the "
                "recording allows: run it with nothing else heavy running (a laptop on AC power, the Jetson in its "
                "maximum power mode); this is reported, not checked"]
    return []
