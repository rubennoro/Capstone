"""UMI's lost-frame rules at 30 fps (``umiproc.run.rules``): every boundary, the split into pieces as UMI's
``06_generate_dataset_plan.py`` does it, and the thresholds as the same times as UMI's at 60 fps."""

import numpy as np
import pytest

from umiproc.run import rules as R

T0 = 5_000_000_000


def _times(n, fps=30.0):
    return (T0 + np.arange(n) * (1e9 / fps)).astype(np.int64)


def _tracked(n, lost=()):
    tr = np.ones(n, bool)
    tr[list(lost)] = False
    return tr


def test_thresholds_are_umis_times_at_30_fps():
    assert R.MAX_LOST_FRAMES / R.FPS == 60 / R.UMI_FPS            # 03_batch_slam.py:44, 1 s
    assert R.MAX_LOST_IN_DEMO / R.FPS == 10 / R.UMI_FPS           # 06:605, 0.17 s
    assert R.MIN_TRACKED_FRAMES / R.FPS == 60 / R.UMI_FPS         # 06:611, 1 s
    assert R.MIN_PIECE_FRAMES / R.FPS == 24 / R.UMI_FPS           # 06:88 min_episode_length, 0.4 s


def test_bool_segments_is_umis_get_bool_segments():
    # 06:41-54 on [T T F T F F T]
    assert R.bool_segments([1, 1, 0, 1, 0, 0, 1]) == [(0, 2, True), (2, 3, False), (3, 4, True), (4, 6, False),
                                                      (6, 7, True)]
    assert R.bool_segments([]) == []


def test_a_clean_demo_is_one_piece():
    r = R.apply_rules(_tracked(60), _times(60), stopped=False)
    assert r.ok and r.reasons == [] and [(p.start, p.stop, p.index) for p in r.pieces] == [(0, 60, 1)]
    assert r.pieces[0].t_first_ns == T0 and r.n_lost == 0 and r.first_tracked_s == 0.0
    assert "1 piece(s) of 60 frames" in r.message


@pytest.mark.parametrize("n_lost, ok", [(5, True), (6, False)])
def test_more_than_5_lost_frames_drop_the_demo(n_lost, ok):
    lost = [12 + 13 * k for k in range(n_lost)]                  # single lost frames between 12-frame runs
    r = R.apply_rules(_tracked(90, lost), _times(90), stopped=False)
    assert ("too_many_lost" in r.reasons) is (not ok)
    if ok:
        assert r.ok and r.n_lost == 5 and len(r.pieces) == 6     # 12-frame pieces between the losses are kept


@pytest.mark.parametrize("n, ok", [(30, True), (29, False)])
def test_fewer_than_30_tracked_frames_drop_the_demo(n, ok):
    r = R.apply_rules(_tracked(n), _times(n), stopped=False)
    assert r.ok is ok and (("too_few_tracked" in r.reasons) is (not ok))


def test_the_demo_is_split_at_its_lost_frames_and_short_pieces_dropped():
    # 60 frames, lost at 10 and 48-49: runs [0,10) (10 < 12: dropped), [11,48) (37), [50,60) (10: dropped)
    r = R.apply_rules(_tracked(60, [10, 48, 49]), _times(60), stopped=False)
    assert r.ok and [(p.start, p.stop) for p in r.pieces] == [(11, 48)]
    assert r.short == [(0, 10), (50, 60)] and r.lost_runs == [(10, 11), (48, 50)]
    # two long runs: two pieces, numbered in order
    r = R.apply_rules(_tracked(60, [29, 30]), _times(60), stopped=False)
    assert [(p.index, p.start, p.stop, p.n_frames) for p in r.pieces] == [(1, 0, 29, 29), (2, 31, 60, 29)]
    # a run of exactly 12 frames is kept, 11 is not (06:725 "< min_episode_length")
    r = R.apply_rules(_tracked(42, [12, 30]), _times(42), stopped=False)
    assert [(p.start, p.stop) for p in r.pieces] == [(0, 12), (13, 30)] and r.short == [(31, 42)]


def test_no_piece_long_enough():
    # 4 lost frames cut 47 frames into runs of 11 (and 3): 43 tracked >= 30, 4 lost <= 5, but no piece
    r = R.apply_rules(_tracked(47, [11, 23, 35, 46]), _times(47), stopped=False)
    assert not r.ok and r.reasons == ["no_piece"] and r.pieces == []


def test_a_late_relocalization_drops_the_demo():
    # the first tracked frame 0.6 s after the first frame (a gap in the frame times, so only 1 frame is lost)
    t = _times(40)
    t[1:] += 600_000_000 - 33_333_333
    r = R.apply_rules(_tracked(40, [0]), t, stopped=False)
    assert not r.ok and r.reasons == ["late_relocalization"] and r.first_tracked_s == pytest.approx(0.6)
    # at exactly 0.5 s it is fine
    t = _times(40)
    t[1:] += 500_000_000 - 33_333_333
    assert R.apply_rules(_tracked(40, [0]), t, stopped=False).ok
    # never tracked
    r = R.apply_rules(np.zeros(40, bool), _times(40), stopped=False)
    assert set(r.reasons) == {"too_many_lost", "too_few_tracked", "late_relocalization", "no_piece"}
    assert r.first_tracked_s is None and "first tracked at never" in r.message


def test_late_start_is_mostly_the_lost_frame_rule():
    # 16 frames lost at the start (0.53 s at 30 fps): both reasons
    r = R.apply_rules(_tracked(60, range(16)), _times(60), stopped=False)
    assert r.reasons == ["too_many_lost", "late_relocalization"]
    # 6 frames lost at the start (0.2 s): only the lost-frame rule
    r = R.apply_rules(_tracked(60, range(6)), _times(60), stopped=False)
    assert r.reasons == ["too_many_lost"]


def test_a_run_stopped_at_30_lost_frames_is_dropped_whatever_it_tracked():
    tr = _tracked(50, range(20, 50))                # 50 frames fed, the last 30 not tracked: the run stopped there
    r = R.apply_rules(tr, _times(90), stopped=True)
    assert not r.ok and r.reasons == ["stopped_lost"] and r.stopped
    assert r.n_frames == 90 and r.n_fed == 50 and r.n_tracked == 20 and r.n_lost == 70    # never-fed frames are lost
    assert "stopped after 50 of 90 frames" in r.message


def test_to_json_has_the_limits():
    j = R.apply_rules(_tracked(60, [30]), _times(60), stopped=False).to_json()
    assert j["limits"]["max_lost_in_demo"] == 5 and j["limits"]["min_piece_frames"] == 12
    assert j["pieces"][0] == {"index": 1, "frames": [0, 30], "n_frames": 30, "t_first_ns": T0,
                              "t_last_ns": int(_times(60)[29]), "seconds": pytest.approx(29 / 30)}
