# adapted from rnoro/final-deployment:deployment/policy_deploy/tests/frames/test_pipeline.py @ 2ffe641
# changes: (+ pi_to_local from tests/frames_pipeline.py) the whole process against the fake Pi through R1 and U2 instead of C1: an episode is recorded and its files
# checked (every .mkv frame the very JPEG the fake Pi sent, video_index = row, times within 2 ms of the truth, no
# ultrawide frame twice, the summary against the CSVs); the dropped-packet, health and stats cases through R1; a
# recording open at shutdown is closed; the v2 sizes; the fatal rig case unchanged.
"""The frames process inside a test (FramesApp with --mock settings), recording through R1 (STRUCTURE.md 5)."""

from __future__ import annotations

import pytest
from frames_helpers import V2_RIG, Running, read_csv, read_mkv, wait_for, write_rig

from teleop_collect.frames.errors import FatalError
from teleop_collect.frames.recorder import FRAME_COLUMNS, PACKET_COLUMNS, RECORDING_FILES


def pi_to_local(fake, t_pi_ns):
    """Invert the fake Pi's clock (a straight line)."""
    a0 = fake.pi_clock_ns(0)
    slope = (fake.pi_clock_ns(10**12) - a0) / 1e12
    return (t_pi_ns - a0) / slope


def record(run, folder, frames=20, timeout_s=15.0, during=None):
    folder.mkdir(parents=True, exist_ok=True)
    ack = run.start_recording(folder)
    assert ack.ok, ack.reason
    if during is not None:
        during()
    wait_for(lambda: run.health().frames_main >= frames, timeout_s, f"{frames} main frames written")
    return run.stop_recording()


def check_episode(run, folder, summary):
    """Every file against what the fake Pi sent; returns the main rows."""
    rec = run.app.fake_pi.recording
    assert sorted(p.name for p in folder.iterdir()) == sorted(RECORDING_FILES)
    rows = {}
    for cam in ("main", "ultrawide"):
        head, rs = read_csv(folder / f"{cam}_frames.csv")
        assert tuple(head) == FRAME_COLUMNS
        frames = rec.cameras[cam].frames
        video = read_mkv(folder / f"{cam}.mkv")
        assert len(video) == len(rs) == getattr(summary, cam).frames
        for i, (r, data) in enumerate(zip(rs, video)):
            assert int(r["video_index"]) == i
            assert data == frames[int(r["frame_no"]) % len(frames)].data        # the bytes the Pi sent
            assert int(r["bytes"]) == len(data)
            assert abs(int(r["t_mid_ns"]) - pi_to_local(run.app.fake_pi, int(r["t_mid_pi_ns"]))) < 2_000_000
        nos = [int(r["frame_no"]) for r in rs]
        assert nos == sorted(set(nos))                                          # never a frame twice
        assert sum(len(d) for d in video) == getattr(summary, cam).bytes
        rows[cam] = rs
    head, pk = read_csv(folder / "packets.csv")
    assert tuple(head) == PACKET_COLUMNS and len(pk) == summary.packets == summary.main.frames
    assert [r["packet_no"] for r in pk] == [r["packet_no"] for r in rows["main"]]
    assert sum(r["uw_reused"] == "1" for r in pk) == summary.uw_reused
    assert sum(r["pair_used"] == "0" for r in pk) == summary.pair_refused
    _, clock = read_csv(folder / "clock_pi.csv")
    assert clock and all(r["offset_ns"] for r in clock)
    assert summary.pi_clock_offset_ns is not None and summary.error == ""
    return rows


def test_ready_record_an_episode_and_its_files(running, tmp_path):
    run = running()
    run.wait_ready()
    summary = record(run, tmp_path / "episode_000", frames=25)
    rows = check_episode(run, tmp_path / "episode_000", summary)
    assert summary.packet_no_gaps == 0 and summary.main.frame_no_gaps == 0
    assert summary.main.first_t_mid_ns == int(rows["main"][0]["t_mid_ns"])
    assert summary.main.last_t_mid_ns == int(rows["main"][-1]["t_mid_ns"])
    summary2 = record(run, tmp_path / "episode_001", frames=10)              # the next one, same process
    check_episode(run, tmp_path / "episode_001", summary2)
    assert int(read_csv(tmp_path / "episode_001" / "packets.csv")[1][0]["packet_no"]) > \
        int(rows["main"][-1]["packet_no"])


def test_v2_sizes_end_to_end(running, v2_episode, tmp_path):
    run = running(v2_episode, rig_file=str(V2_RIG))
    run.wait_ready()
    summary = record(run, tmp_path / "ep", frames=20)
    check_episode(run, tmp_path / "ep", summary)


def test_dropped_packets_show_in_the_summary(running, tmp_path):
    run = running()
    run.wait_ready()
    def drop():                                   # after the recording's first packets, so the gap is inside it
        wait_for(lambda: run.health().frames_main >= 3, 10, "3 frames written")
        run.app.fake_pi.drop_packets(2)
    summary = record(run, tmp_path / "ep", frames=25, during=drop)
    check_episode(run, tmp_path / "ep", summary)
    assert summary.packet_no_gaps == 2 and summary.main.frame_no_gaps == 2


def test_health_ok_stale_ok_by_pause_and_down_by_stop_while_recording(running, tmp_path):
    run = running()
    run.wait_ready()
    (tmp_path / "ep").mkdir()
    assert run.start_recording(tmp_path / "ep").ok
    run.app.fake_pi.pause("main")
    wait_for(lambda: run.health().health.main.status == "stale", 5, "main stale")
    run.app.fake_pi.resume("main")
    wait_for(lambda: run.health().health.main.status == "ok", 5, "main ok again")
    run.app.fake_pi.stop()
    wait_for(lambda: run.health().health.main.status == "down" and run.health().health.ultrawide.status == "down",
             5, "both down")
    fh = run.health()
    assert fh.recording and fh.recording_error == ""
    summary = run.stop_recording()
    assert summary.main.frames > 0 and summary.error == ""


def test_stats_line_shows_the_recording(running, tmp_path):
    run = running()
    run.wait_ready()
    (tmp_path / "ep").mkdir()
    assert run.start_recording(tmp_path / "ep").ok
    wait_for(lambda: run.health().frames_main >= 3, 10, "frames written")
    line = run.app.stats_line(run.app.stats(run.app._clock()))
    assert "recording queue" in line and "written main" in line and "requests r1" in line
    run.stop_recording()
    assert "recording no" in run.app.stats_line(run.app.stats(run.app._clock()))
    wait_for(lambda: (run.app.out_dir / "stats.jsonl").exists(), 5, "stats.jsonl")


def test_a_recording_open_at_shutdown_is_closed(running, tmp_path):
    run = running()
    run.wait_ready()
    (tmp_path / "ep").mkdir()
    assert run.start_recording(tmp_path / "ep").ok
    wait_for(lambda: run.health().frames_main >= 5, 10, "frames written")
    run.stop()
    assert not run.app.recorder.recording
    n = len(read_mkv(tmp_path / "ep" / "main.mkv"))
    assert n >= 5 and len(read_csv(tmp_path / "ep" / "main_frames.csv")[1]) == n


def test_wrong_sensor_in_the_rig_is_fatal(tmp_path, pi_episode):
    def swap(d):
        d["cameras"]["main"]["sensor"], d["cameras"]["ultrawide"]["sensor"] = "imx219", "imx519"
    rig = write_rig(tmp_path, "rig_swapped", swap)
    run = Running(tmp_path / "run", pi_episode, rig_file=str(rig))
    run.thread.join(15)
    assert isinstance(run.error, FatalError) and "rig file says imx219" in str(run.error)
    with pytest.raises(AssertionError):
        run.wait_ready(1.0)
