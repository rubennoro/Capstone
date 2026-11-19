"""FrameRecorder: the episode's camera files from RecordItems (STRUCTURE.md 5.2, PLAN 6.2). New."""

from __future__ import annotations

import threading
import time

import pytest
from deploy_contracts import decode, encode
from frames_helpers import MAIN_SIZE, UW_SIZE, raw_frame, read_csv, read_mkv, wait_for

from teleop_collect.contracts import ClockMap, ClockSample, RecordingSummary
from teleop_collect.frames import recorder as recorder_mod
from teleop_collect.frames.errors import RecordingRefused
from teleop_collect.frames.recorder import (CLOCK_COLUMNS, FRAME_COLUMNS, PACKET_COLUMNS, RECORDING_FILES,
                                            FrameRecorder, RecordItem)

OFFSET = 1_000_000_000_000
P = 33_333_333
T0 = 2_000_000_000_000
#: The files teleop writes into the episode folder before frames starts (PLAN 6.1).
TELEOP_FILES = ("robot_states.csv", "commands.csv", "keys.csv", "clock_pc.csv", "events.jsonl")


def synced_map(now=T0):
    m = ClockMap(60.0, 0.3, 10.0, 500.0, 1e6)
    m.add_sample(ClockSample(t_send_ns=now, t_remote_ns=now + OFFSET, t_recv_ns=now))
    return m


def make_recorder(queue_max=90, clock_every_s=1.0):
    return FrameRecorder(queue_max, clock_every_s, {"main": MAIN_SIZE, "ultrawide": UW_SIZE}, 30, synced_map())


def item(i, uw_no=None, reused=False, pair_used=True, gap_ms=-16.25, exposure_ns=8_000_000):
    """Packet i: main frame i; a new ultrawide frame uw_no unless reused (then the pair is uw_no, not carried)."""
    uw_no = i if uw_no is None else uw_no
    t_main = OFFSET + T0 + i * P
    main = raw_frame("main", i, t_main, exposure_ns=exposure_ns)
    uw = None if reused else raw_frame("ultrawide", uw_no, t_main + 16 * 1_000_000, exposure_ns=4_000_000)
    mid = t_main - exposure_ns // 2
    uw_mid = None if uw is None else uw.info.t_pi_ns - 2_000_000
    return RecordItem(packet_no=i, t_send_pi_ns=t_main + 45_800_000, arrival_ns=T0 + i * P + 47_000_000, main=main,
                      main_t_mid_pi_ns=mid, main_t_mid_ns=mid - OFFSET, ultrawide=uw, ultrawide_t_mid_pi_ns=uw_mid,
                      ultrawide_t_mid_ns=None if uw_mid is None else uw_mid - OFFSET, ultrawide_frame_no=uw_no,
                      uw_reused=reused, pair_used=pair_used, pair_gap_ms=gap_ms)


@pytest.fixture
def ep(tmp_path):
    d = tmp_path / "episode_000"
    d.mkdir()
    return d


def test_rows_per_camera_bytes_columns_and_video_index(ep):
    rec = make_recorder()
    rec.start(ep)
    items = [item(0), item(1), item(2, uw_no=1, reused=True), item(3, uw_no=1, reused=True, pair_used=False,
                                                                       gap_ms=83.0), item(5, uw_no=4)]
    for it in items:
        rec.enqueue(it)
    s = rec.stop()
    assert not rec.recording
    assert sorted(p.name for p in ep.iterdir()) == sorted(RECORDING_FILES)
    head, main_rows = read_csv(ep / "main_frames.csv")
    assert tuple(head) == FRAME_COLUMNS
    assert [int(r["video_index"]) for r in main_rows] == list(range(5))
    assert read_mkv(ep / "main.mkv") == [it.main.data for it in items]
    r = main_rows[1]
    it = items[1]
    assert (int(r["frame_no"]), int(r["t_sensor_pi_ns"]), int(r["exposure_us"]), int(r["t_mid_pi_ns"]),
            int(r["t_mid_ns"]), int(r["packet_no"]), int(r["arrival_ns"]), int(r["bytes"])) == (
        1, it.main.info.t_pi_ns, 8000, it.main_t_mid_pi_ns, it.main_t_mid_ns, 1, it.arrival_ns, len(it.main.data))
    _, uw_rows = read_csv(ep / "ultrawide_frames.csv")
    assert [int(r["frame_no"]) for r in uw_rows] == [0, 1, 4]             # the reused frame 1 written once
    assert [int(r["video_index"]) for r in uw_rows] == [0, 1, 2]
    assert [int(r["packet_no"]) for r in uw_rows] == [0, 1, 5]
    assert read_mkv(ep / "ultrawide.mkv") == [items[k].ultrawide.data for k in (0, 1, 4)]
    head, pk = read_csv(ep / "packets.csv")
    assert tuple(head) == PACKET_COLUMNS
    assert [(r["packet_no"], r["main_frame_no"], r["ultrawide_frame_no"], r["uw_reused"], r["pair_used"])
            for r in pk] == [("0", "0", "0", "0", "1"), ("1", "1", "1", "0", "1"), ("2", "2", "1", "1", "1"),
                             ("3", "3", "1", "1", "0"), ("5", "5", "4", "0", "1")]
    assert float(pk[3]["pair_gap_ms"]) == 83.0 and int(pk[0]["t_send_pi_ns"]) == items[0].t_send_pi_ns
    assert int(pk[0]["arrival_ns"]) == items[0].arrival_ns
    head, clock = read_csv(ep / "clock_pi.csv")
    assert tuple(head) == CLOCK_COLUMNS and len(clock) >= 2               # one at the start, one at the end
    assert int(clock[0]["offset_ns"]) == OFFSET and int(clock[0]["samples"]) == 1
    assert (s.packets, s.packet_no_gaps, s.uw_reused, s.pair_refused, s.error) == (5, 1, 2, 1, "")
    assert (s.main.frames, s.main.frame_no_gaps, s.main.bytes) == (5, 1, sum(len(i.main.data) for i in items))
    assert (s.ultrawide.frames, s.ultrawide.frame_no_gaps) == (3, 2)
    assert s.main.first_t_mid_ns == items[0].main_t_mid_ns and s.main.last_t_mid_ns == items[-1].main_t_mid_ns
    assert s.ultrawide.last_t_mid_ns == items[-1].ultrawide_t_mid_ns
    assert s.pi_clock_offset_ns == OFFSET and s.pi_clock_drift_ppm == 0.0
    assert decode(encode(s), RecordingSummary).main.frames == 5           # passes its contract


def test_an_empty_recording_is_valid(ep):
    rec = make_recorder()
    rec.start(ep)
    s = rec.stop()
    assert s.packets == 0 and s.main.frames == 0 and s.main.first_t_mid_ns is None and s.error == ""
    assert read_csv(ep / "packets.csv")[1] == []
    decode(encode(s), RecordingSummary)


def test_start_refusals(tmp_path, ep):
    rec = make_recorder()
    with pytest.raises(RecordingRefused, match=r"missing: not a folder$"):
        rec.start(tmp_path / "missing")
    (tmp_path / "file").write_text("x")
    with pytest.raises(RecordingRefused, match="not a folder"):
        rec.start(tmp_path / "file")
    rec.start(ep)
    with pytest.raises(RecordingRefused, match="^already recording$"):
        rec.start(tmp_path)
    rec.stop()
    with pytest.raises(RecordingRefused, match="already has recording files$"):
        rec.start(ep)
    other = tmp_path / "other"
    other.mkdir()
    for name in TELEOP_FILES:                       # teleop opens its recorders before R1 start_recording
        (other / name).write_text("teleop's file\n")
    rec.start(other)
    rec.stop()
    assert all((other / name).read_text() == "teleop's file\n" for name in TELEOP_FILES)   # left untouched
    for name in RECORDING_FILES:                    # each one of frames' own files alone is refused
        d = tmp_path / f"has_{name}"
        d.mkdir()
        (d / name).write_text("x")
        with pytest.raises(RecordingRefused, match="already has recording files$"):
            rec.start(d)


def test_a_folder_that_cannot_be_written_is_refused_and_left_clean(tmp_path, monkeypatch):
    d = tmp_path / "ro"
    d.mkdir()

    def broken_open(self, s):
        (s.folder / "main.mkv").write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(FrameRecorder, "_open", broken_open)
    rec = make_recorder()
    with pytest.raises(RecordingRefused, match="cannot create the recording files"):
        rec.start(d)
    assert list(d.iterdir()) == [] and not rec.recording


def test_not_recording_enqueue_is_ignored_and_stop_raises(ep):
    rec = make_recorder()
    rec.enqueue(item(0))
    assert not rec.recording and rec.error == "" and rec.frames_written("main") == 0
    with pytest.raises(RuntimeError):
        rec.stop()


def test_stop_drains_every_queued_item(ep):
    rec = make_recorder(queue_max=200)
    rec.start(ep)
    items = [item(i) for i in range(150)]
    for it in items:
        rec.enqueue(it)
    s = rec.stop()
    assert s.packets == 150 and s.error == ""
    assert read_mkv(ep / "main.mkv") == [it.main.data for it in items]


def test_a_full_queue_fails_the_recording_and_stops_accepting(ep, monkeypatch):
    gate = threading.Event()
    real = FrameRecorder._write

    def slow(self, s, it):
        gate.wait(10)
        real(self, s, it)
    monkeypatch.setattr(FrameRecorder, "_write", slow)
    rec = make_recorder(queue_max=3)
    rec.start(ep)
    t0 = time.monotonic()
    for i in range(10):
        rec.enqueue(item(i))                    # never blocks
    assert time.monotonic() - t0 < 0.5
    assert rec.error.startswith("recording could not keep up: the write queue is full (3 packets)")
    rec.enqueue(item(20))                       # ignored after the failure
    gate.set()
    s = rec.stop()
    assert s.error.startswith("recording could not keep up") and s.packets <= 1
    assert not rec.recording


def test_a_write_error_names_the_file_and_stop_still_closes(ep, monkeypatch):
    calls = {"n": 0}
    real = recorder_mod.MkvWriter.write

    def failing(self, jpeg):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError(28, "No space left on device")
        return real(self, jpeg)
    monkeypatch.setattr(recorder_mod.MkvWriter, "write", failing)
    rec = make_recorder()
    rec.start(ep)
    for i in range(6):
        rec.enqueue(item(i))
    wait_for(lambda: rec.error != "", 5, "the write error")
    assert rec.error.startswith(f"disk error: {ep / 'main.mkv'}:") and "No space left" in rec.error
    s = rec.stop()
    assert s.error == rec.error or s.error.startswith("disk error")
    assert s.packets == 1                       # packet 0 complete; packet 1's main frame failed
    _, pk = read_csv(ep / "packets.csv")
    assert len(pk) == 1


def test_csv_files_are_flushed_while_recording(ep):
    rec = make_recorder(clock_every_s=0.2)
    rec.start(ep)
    for i in range(5):
        rec.enqueue(item(i))
    wait_for(lambda: len(read_csv(ep / "packets.csv")[1]) == 5, 2.0, "packets.csv flushed")
    assert len(read_csv(ep / "main_frames.csv")[1]) == 5
    wait_for(lambda: len(read_csv(ep / "clock_pi.csv")[1]) >= 3, 2.0, "clock rows every 0.2 s")
    assert rec.frames_written("main") == 5 and rec.frames_written("ultrawide") == 5
    rec.stop()


def test_stop_returns_when_the_writer_thread_died(ep, monkeypatch):
    """A writer that died (here: it never ran) must not hang stop(): the main thread keeps serving R1/U2."""
    monkeypatch.setattr(FrameRecorder, "_writer", lambda self, s: None)
    rec = make_recorder(queue_max=3)
    rec.start(ep)
    session = rec._session
    session.thread.join(2)
    for i in range(10):
        rec.enqueue(item(i))                      # the queue fills: nobody takes from it
    t0 = time.monotonic()
    s = rec.stop()
    assert time.monotonic() - t0 < 0.5 and not rec.recording
    assert s.error.startswith("recording could not keep up")          # the first failure is kept
    assert all(fh.closed for fh in session.csv.values()) and all(v.closed for v in session.videos.values())
    (ep.parent / "next").mkdir()
    monkeypatch.undo()
    rec.start(ep.parent / "next")                 # the recorder is usable again
    rec.enqueue(item(0))
    assert rec.stop().packets == 1


def test_stop_returns_when_the_writer_is_stuck(ep, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(recorder_mod, "STOP_PUT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(recorder_mod, "STOP_JOIN_TIMEOUT_S", 0.3)
    monkeypatch.setattr(FrameRecorder, "_write", lambda self, s, it: gate.wait(30))
    rec = make_recorder(queue_max=3)
    rec.start(ep)
    for i in range(10):
        rec.enqueue(item(i))
    t0 = time.monotonic()
    s = rec.stop()
    assert time.monotonic() - t0 < 1.0 and not rec.recording and s.error
    gate.set()                                    # let the daemon thread end


@pytest.mark.parametrize("stage", ["stuck put", "stuck join"])
def test_a_stuck_writer_is_the_recordings_error(ep, monkeypatch, stage):
    gate = threading.Event()
    monkeypatch.setattr(recorder_mod, "STOP_PUT_TIMEOUT_S", 0.2)
    monkeypatch.setattr(recorder_mod, "STOP_JOIN_TIMEOUT_S", 0.2)
    started = threading.Event()

    def blocked(self, s, it):
        started.set()
        gate.wait(30)
    monkeypatch.setattr(FrameRecorder, "_write", blocked)
    rec = make_recorder(queue_max=3)
    rec.start(ep)
    rec.enqueue(item(0))
    started.wait(5)
    if stage == "stuck put":
        for i in range(1, 4):                     # the queue is full, but nothing failed yet
            rec.enqueue(item(i))
        assert rec.error == ""
    s = rec.stop()
    expected = "the writer is stuck" if stage == "stuck put" else "the writer did not finish within 0.2 s"
    assert s.error.startswith(expected), s.error
    gate.set()


def test_any_exception_in_a_write_fails_the_recording_not_the_writer(ep, monkeypatch):
    def broken(self, jpeg):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(recorder_mod.MkvWriter, "write", broken)
    rec = make_recorder()
    rec.start(ep)
    rec.enqueue(item(0))
    wait_for(lambda: rec.error != "", 5, "the error")
    assert rec.error.startswith(f"internal error writing {ep / 'main.mkv'}: RuntimeError: unexpected")
    assert rec._session.thread.is_alive()         # the writer keeps going (discarding) until stop
    rec.enqueue(item(1))                          # ignored
    s = rec.stop()
    assert s.packets == 0 and s.error.startswith("internal error writing")
