import pytest

from fakes import health
from message_comms.episode_loop import Phase
from message_comms.status_line import HOMING_LINE, StatusLine, StatusSnapshot


def snap(**kw):
    values = dict(phase=Phase.RUNNING, episode_id=3, episode_s=12.4, steps_sent=182, cycle_ms=118.0, policy_ms=96.0,
                  image_age_ms=74.0, margin_ms=38.0, waiting_policy_ms=None, skips={}, late=0, anchor_fallbacks=0,
                  odd_chunks=0, skipping_s=None, skipping_reason="", shortened=4, pulled_back=1, loop_late=0,
                  width_clips=0, cameras=health(), arm_mode="running", anchor_dev_mm=1.0)
    values.update(kw)
    return StatusSnapshot(**values)


def fmt(**kw):
    return StatusLine().format(snap(**kw))


def test_idle():
    line = fmt(phase=Phase.IDLE, arm_mode="stopped")
    assert line == "[idle] arm stopped | cams ok 30/ok 30 fps | late loops 0 | keys: c start  h home  q quit"


def test_running_line():
    line = fmt(shortened=4, pulled_back=1)
    assert line == ("[ep 3 running 12.4 s] step 182 | cycle 118 ms (policy 96) | img 74 ms | margin +38 ms | "
                    "short 4 pulled 1 | keys: y n s q")


@pytest.mark.parametrize("field, text", [("shortened", "short 4"), ("pulled_back", "pulled 4"),
                                         ("width_clips", "clip 4"), ("odd_chunks", "odd 4"),
                                         ("loop_late", "late loops 4")])
def test_running_shows_each_counter(field, text):
    line = fmt(**{"shortened": 0, "pulled_back": 0, field: 4})
    assert text in line and line.endswith("keys: y n s q") and len(line) <= 120


def test_running_all_counters_fit():
    line = fmt(shortened=4, pulled_back=1, width_clips=2, odd_chunks=3, loop_late=5)
    assert len(line) <= 120 and "margin +38 ms" in line and "short 4" in line and "odd 3" in line


def test_negative_margin():
    assert "margin -12 ms" in fmt(margin_ms=-12.0)


def test_waiting_for_policy():
    assert fmt(waiting_policy_ms=1900.0, arm_mode="holding") == \
        "[ep 3 running 14.0 s] waiting for policy 1.9 s | arm holding | keys: y n s q".replace("14.0", "12.4")


def test_skipping():
    line = fmt(episode_s=31.0, skipping_s=12.4, skipping_reason="main camera stale", arm_mode="holding")
    assert line == "[ep 3 running 31.0 s] SKIPPING 12.4 s: main camera stale | arm holding | keys: y n s q"


def test_homing():
    assert fmt(phase=Phase.HOMING) == HOMING_LINE
    assert "keys inactive; Ctrl-C quits; immediate stop: the robot's external stop button" in HOMING_LINE


@pytest.mark.parametrize("kw", [
    dict(),
    dict(late=99, anchor_fallbacks=12, skips={"camera_stale": 400}, shortened=12345, pulled_back=9999,
         width_clips=777, odd_chunks=55, loop_late=123456, steps_sent=99999, cycle_ms=12345.0, margin_ms=-9999.0),
    dict(skipping_s=1e6, skipping_reason="x" * 300),
    dict(phase=Phase.IDLE, arm_mode="y" * 200),
    dict(phase=Phase.RESETTING),
])
def test_never_longer_than_120(kw):
    assert len(fmt(**kw)) <= 120
