"""umiproc.slam.log: the UMIPROC_SUMMARY line and the fork's stdout events, on recorded logs and synthetic text."""

import json
from pathlib import Path

import pytest

from umiproc.slam.log import (SUMMARY_FORMAT, SUMMARY_PREFIX, MapInfo, Pacing, SlamLogError, parse_events,
                              parse_log, parse_summary, read_log)

FIX = Path(__file__).parent / "fixtures" / "slam"


def _summary(**over) -> dict:
    d = {"format": SUMMARY_FORMAT, "exit_code": 0, "error": None, "frames_total": 10, "frames_fed": 10,
         "lost_frames": 2, "not_tracking_frames": 1, "stopped_early": False, "stop_reason": None,
         "state_counts": {"2": 8, "3": 1, "6": 1},
         "maps": [{"id": 0, "keyframes": 5, "map_points": 100, "imu_initialized": True, "inertial_ba1": True,
                   "inertial_ba2": False}],
         "t0_ns": 123, "map_removed": False, "options": {"init_tag_id": 13},
         "pacing": {"track_ms": {"median": 21.5, "p95": 30.25, "max": 41.0}, "frames_over_period": 2,
                    "feed_rate_hz": 29.5, "recording_rate_hz": 29.98, "feed_wall_s": 0.339}}
    d.update(over)
    return d


def _line(d: dict) -> str:
    return SUMMARY_PREFIX + json.dumps(d)


# ---------------------------------------------------------------------------------------------- recorded logs

def test_recorded_map_run_reset_at_end():
    # mapping run whose last map was reset just before the end: no map left, five two-view initialisations
    lg = read_log(FIX / "map_run_reset_at_end.log")
    s = lg.summary
    assert (s.frames_total, s.frames_fed, s.lost_frames, s.not_tracking_frames) == (733, 733, 397, 392)
    assert s.state_counts == {1: 392, 2: 336, 3: 5}
    assert s.maps == () and s.maps_with_keyframes == ()
    assert s.t0_ns == 13676390000 and s.exit_code == 0 and s.options["init_tag_id"] == 13
    assert [e.detail["method"] for e in lg.of_kind("map_initialized")] == ["two_view"] * 5
    assert [e.detail["reason"] for e in lg.resets] == ["lost_before_viba2"] * 5
    assert lg.viba_completed == (1,)                 # VIBA 1 ran on a map that was reset later
    assert lg.of_kind("relocalized") == []


def test_recorded_map_run():
    lg = read_log(FIX / "map_run.log")
    s = lg.summary
    assert s.maps == (MapInfo(id=0, keyframes=34, map_points=1228, imu_initialized=True, inertial_ba1=False,
                              inertial_ba2=False),)
    assert len(s.maps_with_keyframes) == 1
    assert len(lg.of_kind("map_initialized")) == 3 and len(lg.resets) == 2
    assert lg.viba_completed == ()


def test_recorded_localize_run():
    lg = read_log(FIX / "localize_run.log")
    s = lg.summary
    assert (s.frames_fed, s.lost_frames, s.not_tracking_frames) == (362, 3, 3)
    assert s.state_counts == {2: 359, 6: 3}
    assert s.options["init_tag_id"] == -1 and s.options["load_map"].endswith(".osa")
    assert lg.relocalization_times_s == [0.0, 6.804559, 10.907304]
    assert [e.detail["keyframe_id"] for e in lg.of_kind("relocalized")] == [100, 141, 163]
    assert lg.viba_completed == (1, 2)
    assert lg.relocalization_failures == 1
    assert lg.resets == []


def test_recorded_v2_summary_with_pacing():
    # the summary line of a real run of the current front-end (localization of episode 119, laptop on battery at
    # 400 MHz: tracking far slower than the camera)
    s = read_log(FIX / "localize_run_v2_pacing.log").summary
    assert (s.frames_fed, s.lost_frames) == (362, 16)
    assert s.pacing == Pacing(track_ms_median=214.256, track_ms_p95=245.827, track_ms_max=343.827,
                              frames_over_period=362, feed_rate_hz=3.702, recording_rate_hz=29.98,
                              feed_wall_s=97.79)


def test_v1_summaries_have_no_pacing():
    # the recorded logs above are from the front-end before the pacing stats (format v1)
    for name in ("map_run.log", "localize_run.log", "map_run_reset_at_end.log"):
        assert read_log(FIX / name).summary.pacing is None
    assert parse_summary(_line(_summary(format="umiproc.slam_summary.v1"))).pacing is None


# ---------------------------------------------------------------------------------------------- summary line

def test_pacing_fields():
    p = parse_summary(_line(_summary())).pacing
    assert (p.track_ms_median, p.track_ms_p95, p.track_ms_max) == (21.5, 30.25, 41.0)
    assert (p.frames_over_period, p.feed_rate_hz, p.recording_rate_hz, p.feed_wall_s) == (2, 29.5, 29.98, 0.339)
    assert parse_summary(_line(_summary(pacing=None, frames_fed=0))).pacing is None      # no frame fed

def test_summary_absent_is_none():
    assert parse_summary("frames 10\nShutdown\n") is None
    assert parse_log("Shutdown\n").summary is None


def test_last_summary_line_wins():
    text = "\n".join([_line(_summary(frames_fed=3)), "noise", _line(_summary(frames_fed=7))])
    assert parse_summary(text).frames_fed == 7


def test_summary_fields():
    s = parse_summary(_line(_summary(maps=None, t0_ns=None, error="cannot read x", exit_code=1,
                                     stop_reason="unreadable_frame", stopped_early=True)))
    assert s.maps is None and s.maps_with_keyframes == () and s.t0_ns is None
    assert (s.exit_code, s.error, s.stop_reason, s.stopped_early) == (1, "cannot read x", "unreadable_frame", True)
    assert s.state_counts == {2: 8, 3: 1, 6: 1}


@pytest.mark.parametrize("bad", [
    SUMMARY_PREFIX + "{not json",
    SUMMARY_PREFIX + "[1, 2]",
    _line(_summary(format="something.else")),
    _line({k: v for k, v in _summary().items() if k != "frames_fed"}),
    _line(_summary(stopped_early=1)),
    _line(_summary(frames_fed=True)),
    _line(_summary(maps=[{"id": 0}])),
    _line({k: v for k, v in _summary().items() if k != "pacing"}),                    # v2 must have it
    _line(_summary(pacing=[1, 2])),
    _line(_summary(pacing={"track_ms": {"median": 1.0, "p95": 2.0}, "frames_over_period": 0, "feed_rate_hz": 1.0,
                           "recording_rate_hz": 30.0, "feed_wall_s": 1.0})),
    _line(_summary(pacing={**_summary()["pacing"], "feed_rate_hz": "fast"})),
    _line(_summary(pacing={**_summary()["pacing"], "frames_over_period": 1.5})),
])
def test_malformed_summary_raises(bad):
    with pytest.raises(SlamLogError):
        parse_summary(bad)


# ---------------------------------------------------------------------------------------------- events

def test_init_method_tag_and_two_view():
    text = "\n".join(["init_success=0", " tag_init_success=1", "First KF:0; Map init KF:0",
                      "New Map created with 50 points", "init_success=1", "New Map created with 60 points",
                      "New Map created with 70 points"])
    ev = parse_log(text).of_kind("map_initialized")
    assert [(e.detail["points"], e.detail["method"]) for e in ev] == [(50, "tag"), (60, "two_view"), (70, None)]


@pytest.mark.parametrize("line,code", [
    ("IMU is not or recently initialized. Reseting active map...", "lost_before_viba2"),
    ("TRACK: Reset map because local mapper set the bad imu flag ", "bad_imu"),
    ("Timestamp jump detected, before IMU initialization. Reseting...", "timestamp_jump"),
    ("Wrong initialization, reseting...", "wrong_initialization"),
    ("Track lost before IMU initialisation, reseting...", "lost_before_imu_init"),
])
def test_reset_reasons(line, code):
    lg = parse_log(f"{line}\nSYSTEM-> Reseting active map in monocular case\n")
    assert [e.detail["reason"] for e in lg.resets] == [code]


def test_reset_without_reason_and_stale_reason_cleared():
    text = "\n".join(["Timestamp jump detected. State set to LOST. Reseting IMU integration...",
                      "Creation of new map with id: 1",               # that jump started a new map, not a reset
                      "SYSTEM-> Reseting active map in monocular case"])
    lg = parse_log(text)
    assert [e.detail for e in lg.of_kind("map_created")] == [{"map_id": 1}]
    assert [e.detail["reason"] for e in lg.resets] == [None]


def test_relocalized_without_details_and_line_numbers():
    text = "x\nINIT_RELOCALIZE success!\nINIT_RELOCALIZE success!\npKFCur->mTimeStamp1.5\npKFcur->mnId7\n"
    ev = parse_log(text).of_kind("relocalized")
    assert [(e.line, e.detail["t_s"], e.detail["keyframe_id"]) for e in ev] == [(2, None, None), (3, 1.5, 7)]


def test_viba_and_stopped_lost():
    text = "start VIBA 1\nend VIBA 1\nstart VIBA 2\nLost tracking on 30 >= 30 frames. Stopping.\n"
    events, n_failed = parse_events(text)
    assert [e.kind for e in events] == ["viba_start", "viba_end", "viba_start", "stopped_lost"]
    assert events[-1].detail == {"lost": 30, "max": 30}
    assert parse_log(text).viba_completed == (1,)
    assert n_failed == 0
