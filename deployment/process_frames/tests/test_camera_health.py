"""CameraHealth: ok / stale / down from frame ages (STRUCTURE.md §8), driven by a fake clock."""

from __future__ import annotations

import pytest

from contracts import CameraHealthReport, HealthReport, HealthReply, decode, encode

from process_frames.camera_health import CameraHealth

MS = 1_000_000


@pytest.fixture
def health(fake_clock):
    return CameraHealth("main", stale_s=0.35, down_s=2.0, fps_window_s=1.0, clock=fake_clock)


def test_stale_before_the_first_frame_then_down(health, fake_clock):
    assert health.status() == "stale"
    fake_clock.advance(2000 * MS)
    assert health.status() == "stale"
    fake_clock.advance(1)
    assert health.status() == "down"


def test_ok_stale_ok_and_ok_down(health, fake_clock):
    health.record_frame(fake_clock.now)
    assert health.status() == "ok"
    fake_clock.advance(350 * MS)
    assert health.status() == "ok"
    fake_clock.advance(1)
    assert health.status() == "stale"
    health.record_frame(fake_clock.now - 10 * MS)
    assert health.status() == "ok"
    fake_clock.advance(2001 * MS)
    assert health.status() == "down"
    health.record_frame(fake_clock.now)                   # down is not latched without mark_down
    assert health.status() == "ok"


def test_age_is_clamped_at_zero(health, fake_clock):
    health.record_frame(fake_clock.now + 3 * MS)          # a few ms "in the future"
    r = health.report()
    assert r.last_frame_age_ms == 0.0 and r.status == "ok"


def test_latched_down_stays(health, fake_clock):
    health.record_frame(fake_clock.now)
    health.mark_down("Pi stream restarted: seq 12 after 9031")
    assert health.status() == "down"
    health.record_frame(fake_clock.now + 1)
    assert health.status() == "down"


def test_fps_over_the_window_and_dropped(health, fake_clock):
    assert health.fps() == 0.0
    for i in range(61):                                   # 2 s at 30 FPS
        fake_clock.now = 1_000_000_000_000 + i * 33_333_333
        health.record_frame(fake_clock.now)
    assert health.fps() == pytest.approx(30.0, rel=0.02)
    fake_clock.advance(3000 * MS)
    assert health.fps() == 0.0                            # nothing in the last second
    health.record_dropped(3)
    health.record_dropped(2)
    assert health.dropped == 5 and health.report().dropped == 5


def test_report_passes_the_contract(health, fake_clock):
    health.record_frame(fake_clock.now - 41 * MS)
    r = health.report()
    assert isinstance(r, CameraHealthReport)
    assert r.status == "ok" and r.last_frame_age_ms == pytest.approx(41.0)
    msg = HealthReply(health=HealthReport(main=r, ultrawide=r))
    back = decode(encode(msg), expect=HealthReply)
    assert back.health.main.status == "ok"
