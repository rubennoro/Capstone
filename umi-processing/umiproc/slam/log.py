"""Parse what ``umiproc_slam`` prints: its ``UMIPROC_SUMMARY {json}`` line and the fork's own stdout events.

The event strings are the exact ones printed by UMI's ORB-SLAM3 fork (b741dca + native/patches); the fork sets the
verbosity to QUIET after start-up, so only ``cout`` lines and QUIET messages appear. Where they come from:

========================  ================================================================  ===========================
event                     line                                                              source
========================  ================================================================  ===========================
map_created               ``Creation of new map with id: N``                                Atlas.cc CreateNewMap
map_initialized           ``New Map created with N points``                                 Tracking.cc (mono init)
  its method              the last ``init_success=1`` (two views) / `` tag_init_success=1``  Tracking.cc
                          (table tag) before it
map_reset                 ``SYSTEM-> Reseting active map in monocular case``                System.cc LocalizeMonocular
  its reason              the last reason line before it (``RESET_REASONS``)                 Tracking.cc
relocalized               ``INIT_RELOCALIZE success!`` then ``pKFCur->mTimeStamp<t>``       Tracking.cc INIT_RELOCALIZE
                          (frame time in s since t0) and ``pKFcur->mnId<id>``
viba_start / viba_end     ``start VIBA 1`` ``end VIBA 1`` ``start VIBA 2`` ``end VIBA 2``      LocalMapping.cc Run
stopped_lost              ``Lost tracking on N >= M frames. Stopping.``                      umiproc_slam
========================  ================================================================  ===========================

The first IMU initialisation prints nothing; ``SlamSummary.maps`` (System::MapSummary after shutdown) says whether
each final map's IMU is initialised and which of the two inertial refinements (VIBA 1, VIBA 2) ran on it. Lines from
the tracking and local-mapping threads interleave, so event order across threads is approximate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SUMMARY_PREFIX = "UMIPROC_SUMMARY "
SUMMARY_FORMAT = "umiproc.slam_summary.v2"
#: Formats read: v1 (before the pacing object; its ``pacing`` is None) and v2 (the current front-end's).
SUMMARY_FORMATS_READ = ("umiproc.slam_summary.v1", SUMMARY_FORMAT)

# Tracking::eTrackingState (include/Tracking.h)
STATE_NAMES = {-1: "SYSTEM_NOT_READY", 0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED", 2: "OK", 3: "RECENTLY_LOST",
               4: "LOST", 5: "OK_KLT", 6: "INIT_RELOCALIZE"}
STATE_OK = 2

# reason line (prefix) -> reason code, for the map reset that follows it
RESET_REASONS = {
    "IMU is not or recently initialized. Reseting active map...": "lost_before_viba2",
    "TRACK: Reset map because local mapper set the bad imu flag": "bad_imu",
    "Timestamp jump detected": "timestamp_jump",
    "Wrong initialization, reseting...": "wrong_initialization",
    "Track lost before IMU initialisation, reseting...": "lost_before_imu_init",
}

_MAP_CREATED = re.compile(r"^Creation of new map with id: (\d+)\s*$")
_MAP_INITIALIZED = re.compile(r"^New Map created with (\d+) points\s*$")
_RESET = "SYSTEM-> Reseting active map in monocular case"
_RELOC = "INIT_RELOCALIZE success!"
_RELOC_TIME = re.compile(r"^pKFCur->mTimeStamp(\S+)\s*$")
_RELOC_KF = re.compile(r"^pKFcur->mnId(\d+)\s*$")
_VIBA = re.compile(r"^(start|end) VIBA ([12])\s*$")
_STOPPED = re.compile(r"^Lost tracking on (\d+) >= (\d+) frames\. Stopping\.\s*$")
_RELOC_FAILED = "Relocalization() failed."
_RELOC_LOOKAHEAD = 20          # lines after "INIT_RELOCALIZE success!" searched for its time and keyframe id


class SlamLogError(ValueError):
    """The summary line is there but is not valid."""


@dataclass(frozen=True)
class MapInfo:
    """One map of the final atlas (System::MapSummary)."""

    id: int
    keyframes: int
    map_points: int
    imu_initialized: bool
    inertial_ba1: bool           # VIBA 1 ran on this map
    inertial_ba2: bool           # VIBA 2 ran on this map


@dataclass(frozen=True)
class Pacing:
    """Whether the machine kept up with the frames (the summary's ``pacing``; native/umi_slam/umiproc_slam.cc)."""

    track_ms_median: float       # per-frame tracking time (IMU batch + LocalizeMonocular), ms
    track_ms_p95: float          # nearest-rank 95th percentile
    track_ms_max: float
    frames_over_period: int      # frames whose tracking took longer than their frame period
    feed_rate_hz: float          # frames fed / wall time of the feeding loop (frame reads and waits included)
    recording_rate_hz: float     # the fed frames' own rate
    feed_wall_s: float


@dataclass(frozen=True)
class SlamSummary:
    """The ``UMIPROC_SUMMARY`` line (native/umi_slam/umiproc_slam.cc)."""

    exit_code: int
    error: str | None
    frames_total: int            # lines in times.txt
    frames_fed: int
    lost_frames: int             # state != OK after the frame
    not_tracking_frames: int     # LocalizeMonocular false (what --max_lost_frames counts)
    stopped_early: bool
    stop_reason: str | None      # "max_lost_frames" | "unreadable_frame" | None
    state_counts: dict[int, int]
    maps: tuple[MapInfo, ...] | None   # None: the SLAM system never ran (bad input)
    t0_ns: int | None
    map_removed: bool
    options: dict
    pacing: Pacing | None = None  # None: no frame was fed, or a v1 summary (front-end built before the pacing stats)

    @property
    def maps_with_keyframes(self) -> tuple[MapInfo, ...]:
        return tuple(m for m in (self.maps or ()) if m.keyframes > 0)


@dataclass(frozen=True)
class SlamEvent:
    kind: str                    # map_created | map_initialized | map_reset | relocalized | viba_start | viba_end |
                                 # stopped_lost
    line: int                    # 1-based line number in the log
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SlamLog:
    summary: SlamSummary | None  # None when the run died before printing it
    events: tuple[SlamEvent, ...]
    relocalization_failures: int  # "Relocalization() failed." lines (one per frame waiting to relocalize)

    def of_kind(self, kind: str) -> list[SlamEvent]:
        return [e for e in self.events if e.kind == kind]

    @property
    def relocalization_times_s(self) -> list[float | None]:
        """Frame time (s since t0) of every INIT_RELOCALIZE success, in order."""
        return [e.detail.get("t_s") for e in self.of_kind("relocalized")]

    @property
    def resets(self) -> list[SlamEvent]:
        return self.of_kind("map_reset")

    @property
    def viba_completed(self) -> tuple[int, ...]:
        """VIBA stages that finished somewhere in the run (also in maps reset later: see SlamSummary.maps)."""
        return tuple(e.detail["stage"] for e in self.of_kind("viba_end"))


def _bool(d: dict, key: str) -> bool:
    v = d[key]
    if not isinstance(v, bool):
        raise SlamLogError(f"summary field {key!r} is not a boolean: {v!r}")
    return v


def _int(d: dict, key: str, *, optional: bool = False) -> int | None:
    v = d[key]
    if v is None and optional:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        raise SlamLogError(f"summary field {key!r} is not an integer: {v!r}")
    return v


def _float(d: dict, key: str) -> float:
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SlamLogError(f"summary field {key!r} is not a number: {v!r}")
    return float(v)


def _pacing(d) -> Pacing | None:
    if d is None:
        return None
    if not isinstance(d, dict):
        raise SlamLogError(f"summary field 'pacing' is not an object: {d!r}")
    t = d["track_ms"]
    return Pacing(track_ms_median=_float(t, "median"), track_ms_p95=_float(t, "p95"), track_ms_max=_float(t, "max"),
                  frames_over_period=_int(d, "frames_over_period"), feed_rate_hz=_float(d, "feed_rate_hz"),
                  recording_rate_hz=_float(d, "recording_rate_hz"), feed_wall_s=_float(d, "feed_wall_s"))


def summary_from_json(d: dict) -> SlamSummary:
    """Build a SlamSummary from the decoded JSON object (raises SlamLogError on a malformed one)."""
    try:
        fmt = d.get("format")
        if fmt not in SUMMARY_FORMATS_READ:
            raise SlamLogError(f"unknown summary format {fmt!r} (want one of {SUMMARY_FORMATS_READ})")
        maps = None
        if d["maps"] is not None:
            maps = tuple(MapInfo(id=_int(m, "id"), keyframes=_int(m, "keyframes"), map_points=_int(m, "map_points"),
                                 imu_initialized=_bool(m, "imu_initialized"), inertial_ba1=_bool(m, "inertial_ba1"),
                                 inertial_ba2=_bool(m, "inertial_ba2")) for m in d["maps"])
        return SlamSummary(
            exit_code=_int(d, "exit_code"), error=d["error"], frames_total=_int(d, "frames_total"),
            frames_fed=_int(d, "frames_fed"), lost_frames=_int(d, "lost_frames"),
            not_tracking_frames=_int(d, "not_tracking_frames"), stopped_early=_bool(d, "stopped_early"),
            stop_reason=d["stop_reason"], state_counts={int(k): int(v) for k, v in d["state_counts"].items()},
            maps=maps, t0_ns=_int(d, "t0_ns", optional=True), map_removed=_bool(d, "map_removed"),
            options=dict(d["options"]), pacing=None if fmt != SUMMARY_FORMAT else _pacing(d["pacing"]))
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        if isinstance(e, SlamLogError):
            raise
        raise SlamLogError(f"malformed summary: {type(e).__name__}: {e}") from e


def parse_summary(text: str) -> SlamSummary | None:
    """The last ``UMIPROC_SUMMARY`` line of a log, or None when there is none."""
    lines = [ln for ln in text.splitlines() if ln.startswith(SUMMARY_PREFIX)]
    if not lines:
        return None
    try:
        d = json.loads(lines[-1][len(SUMMARY_PREFIX):])
    except json.JSONDecodeError as e:
        raise SlamLogError(f"summary line is not JSON: {e}") from e
    if not isinstance(d, dict):
        raise SlamLogError("summary line is not a JSON object")
    return summary_from_json(d)


def _float_or_none(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def parse_events(text: str) -> tuple[tuple[SlamEvent, ...], int]:
    """The fork's events in log order, and the number of failed relocalization attempts."""
    lines = text.splitlines()
    events: list[SlamEvent] = []
    init_method: str | None = None        # from the last init_success / tag_init_success = 1 line
    reset_reason: str | None = None       # from the last reason line
    n_reloc_failed = 0
    for i, raw in enumerate(lines):
        ln = raw.strip()
        if not ln:
            continue
        if ln == _RELOC_FAILED:
            n_reloc_failed += 1
            continue
        if ln == "init_success=1":
            init_method = "two_view"
            continue
        if ln == "tag_init_success=1":
            init_method = "tag"
            continue
        for prefix, code in RESET_REASONS.items():
            if ln.startswith(prefix):
                reset_reason = code
                break
        if m := _MAP_CREATED.match(ln):
            events.append(SlamEvent("map_created", i + 1, {"map_id": int(m.group(1))}))
            reset_reason = None                 # a reason line followed by a new map, not a reset
        elif m := _MAP_INITIALIZED.match(ln):
            events.append(SlamEvent("map_initialized", i + 1, {"points": int(m.group(1)), "method": init_method}))
            init_method = None
            reset_reason = None
        elif ln == _RESET:
            events.append(SlamEvent("map_reset", i + 1, {"reason": reset_reason}))
            reset_reason = None
        elif ln == _RELOC:
            detail: dict = {"t_s": None, "keyframe_id": None}
            for nxt in lines[i + 1:i + 1 + _RELOC_LOOKAHEAD]:
                nxt = nxt.strip()
                if nxt == _RELOC:
                    break
                if detail["t_s"] is None and (mt := _RELOC_TIME.match(nxt)):
                    detail["t_s"] = _float_or_none(mt.group(1))
                elif detail["keyframe_id"] is None and (mk := _RELOC_KF.match(nxt)):
                    detail["keyframe_id"] = int(mk.group(1))
            events.append(SlamEvent("relocalized", i + 1, detail))
        elif m := _VIBA.match(ln):
            events.append(SlamEvent(f"viba_{m.group(1)}", i + 1, {"stage": int(m.group(2))}))
        elif m := _STOPPED.match(ln):
            events.append(SlamEvent("stopped_lost", i + 1, {"lost": int(m.group(1)), "max": int(m.group(2))}))
    return tuple(events), n_reloc_failed


def parse_log(text: str) -> SlamLog:
    events, n_failed = parse_events(text)
    return SlamLog(summary=parse_summary(text), events=events, relocalization_failures=n_failed)


def read_log(path: str | Path) -> SlamLog:
    return parse_log(Path(path).read_text(errors="replace"))
